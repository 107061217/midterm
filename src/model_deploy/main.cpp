#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <cmath>
#include "DA7212.h"
#include "mbed.h"
#include "uLCD_4DGL.h"
DA7212 audio;

DigitalIn sw2(SW2); //mode
DigitalIn sw3(SW3); //enter
uLCD_4DGL uLCD(D1, D0, D2);
Serial pc(USBTX, USBRX);
int16_t waveform[kAudioTxBufferSize];
int output = 0;         // gesture
int play= 0;            // play song or not
static int song = 1;    // song 1, 2, or 3

void SelectMode();
int PredictGesture(float* output);
void DNN();
void playNote(int freq);
void PlaySong();
void loadSignal();
Thread t1 (osPriorityNormal, 120 * 1024);      // for DNN
Thread t2;                                     // for PLAYSONG
Thread t3;                                     // for LOADSIGNAL

// Return the result of the last prediction
int PredictGesture(float* output) {
    // How many times the most recent gesture has been matched in a row
    static int continuous_count = 0;
    // The result of the last prediction
    static int last_predict = -1;

    // Find whichever output has a probability > 0.8 (they sum to 1)
    int this_predict = -1;
    for (int i = 0; i < label_num; i++) {
        if (output[i] > 0.8) this_predict = i;
    }

    // No gesture was detected above the threshold
    if (this_predict == -1) {
        continuous_count = 0;
        last_predict = label_num;
        return label_num;
    }

    if (last_predict == this_predict) {
        continuous_count += 1;
    } 
    else {
        continuous_count = 0;
    }
    last_predict = this_predict;

    // If we haven't yet had enough consecutive matches for this gesture,
    // report a negative result
    if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
        return label_num;
    }
    // Otherwise, we've seen a positive result, so clear all our variables
    // and report it
    continuous_count = 0;
    last_predict = -1;

    return this_predict;
}
void DNN(void) {
    // Create an area of memory to use for input, output, and intermediate arrays.
    // The size of this will depend on the model you're using, and may need to be
    // determined by experimentation.
    constexpr int kTensorArenaSize = 60 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    // Whether we should clear the buffer next time we fetch data
    bool should_clear_buffer = false;
    bool got_data = false;

    // The gesture index of the prediction
    int gesture_index;

    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    static tflite::MicroOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                             tflite::ops::micro::Register_RESHAPE(), 1);
    // Build an interpreter to run the model with
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite::MicroInterpreter* interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    interpreter->AllocateTensors();

    // Obtain pointer to the model's input tensor
    TfLiteTensor* model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
            error_reporter->Report("Bad input tensor parameters in model");
        return;
    }

    int input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk) {
        error_reporter->Report("Set up failed\n");
        return;
    }

    error_reporter->Report("Set up successful...\n");

    while (true) {

        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
            should_clear_buffer = false;
            continue;
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on index: %d\n", begin_index);
            continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        output = 0;
        if (gesture_index < label_num){
            if (gesture_index == 0){
                output = 1;      //ring
            }    
            else if (gesture_index == 1){
                output = 2;     //slope
            } 
               
            else if (gesture_index == 2){
                output = 3;     //M
            }  
        }
    }
}

int song1[42] = {
    261, 261, 392, 392, 440, 440, 392, //1155665
    349, 349, 330, 330, 294, 294, 261, //4433221
    392, 392, 349, 349, 330, 330, 294, //5544332
    392, 392, 349, 349, 330, 330, 294, //5544332
    261, 261, 392, 392, 440, 440, 392, //1155665
    349, 349, 330, 330, 294, 294, 261  //4433221
};
int song2[24] = {
    392, 330, 330, 349, 294, 294,       //533422
    261, 294, 330, 349, 392, 392, 392,  //1234555
    392, 330, 330, 349, 294, 294,       //533422
    261, 330, 392, 392, 261             //13551
};
int song3[32] = {
    261, 294, 330, 261, 261, 294, 330, 261, //12311231
    330, 349, 392, 330, 349, 392,           //345345
    392, 440, 392, 349, 330, 261,           //565431
    392, 440, 392, 349, 330, 261,           //565431
    294, 392, 261, 294, 392, 261            //251251
};
int noteLength1[42] = {
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2
};
int noteLength2[24] {
    1, 1, 1, 1, 1, 1,  
    1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1
}; 
int noteLength3[32] {
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1
};

void playNote(int freq) {
    for(int i = 0; i < kAudioTxBufferSize; i++) {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
 // the loop below will play the note for the duration of 1s

    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j){
        audio.spk.play(waveform, kAudioTxBufferSize);
    }
}
void PlaySong() {
    while(1) {
        if (play == 0){
           playNote(0); 
        }   
        if (song == 1 && play == 1) {
            for(int i = 0; i < 42 && play == 1; i++) {
                int length = noteLength1[i];
                while(length-- && play == 1) {
                // the loop below will play the note for the duration of 1s
                    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize  && play == 1; ++j) {
                        playNote(song1[i]);
                    }
                    if(length < 1) {
                        wait(1.0);
                    }
                }
            }
        }
        else if (song == 2 && play == 1) {
            for(int i = 0; i < 24 && play == 1; i++) {
                int length = noteLength2[i];
                while(length-- && play == 1) {
                // the loop below will play the note for the duration of 1s
                    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize  && play == 1; ++j) {
                        playNote(song2[i]);
                    }
                    if(length < 1){
                        wait(1.0);  
                    } 
                }
            }
        }
        else if (song == 3 && play == 1) {
            for(int i = 0; i < 32 && play == 1; i++) {
                int length = noteLength3[i];
                while(length-- && play == 1) {
                // the loop below will play the note for the duration of 1s
                    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize  && play == 1; ++j) {
                        playNote(song3[i]);
                    }
                    if(length < 1) {
                        wait(1.0);
                    }
                }
            }
        }
    }
}

void SelectMode() {
    play = 0;                
    int mode = 0;
  
    uLCD.cls();
    uLCD.color(WHITE);
    uLCD.printf("\nSelect the Mode\n"); 
    uLCD.color(RED);
    uLCD.printf("Foward\n");
    uLCD.color(GREEN);
    uLCD.printf("Backward\n");
    uLCD.printf("Choose song\n");
    uLCD.printf("Taiko\n");
    while(1) {
        if(output == 1) {      // next mode
            uLCD.cls();
            uLCD.color(WHITE);
            uLCD.printf("\nSelect the Mode\n"); 
            if (mode == 1) {
                uLCD.color(GREEN);
                uLCD.printf("Foward\n");
                uLCD.printf("Backward\n");
                uLCD.color(RED);
                uLCD.printf("Choose song\n");
                uLCD.color(GREEN);
                uLCD.printf("Taiko\n");
                mode = 2;
            }
            else if (mode == 2) {
                uLCD.color(GREEN);
                uLCD.printf("Foward\n");
                uLCD.printf("Backward\n");
                uLCD.printf("Choose song\n");
                uLCD.color(RED);
                uLCD.printf("Taiko\n");
                mode = 3;
            }
            else if (mode == 3) {
                uLCD.color(RED);
                uLCD.printf("Foward\n");
                uLCD.color(GREEN);
                uLCD.printf("Backward\n");
                uLCD.printf("Choose song\n");
                uLCD.printf("Taiko\n");
                mode = 0;
            }
            else if (mode == 0) {
                uLCD.color(GREEN);
                uLCD.printf("Foward\n");
                uLCD.color(RED);
                uLCD.printf("Backward\n");
                uLCD.color(GREEN);
                uLCD.printf("Choose song\n");
                uLCD.printf("Taiko\n");
                mode = 1;
            }
            wait(0.1);
        }
        else if (!sw3) {
            uLCD.cls();
            uLCD.color(GREEN);  //FOWARD
            if (mode == 0) { 
                play = 1;   
                if (song == 1) {
                    uLCD.printf("\nPlaying song 2\n");
                    song = 2;
                    return;
                }
                else if (song == 2) {
                    uLCD.printf("\nPlaying song 3\n");
                    song = 3;
                    return;
                }
                else if (song == 3) {
                    uLCD.printf("\nPlaying song 1\n");
                    song = 1;
                    return;
                }
            }
            else if (mode == 1) { //BACKWARD 
                play = 1;   
                if (song == 1) {
                    uLCD.printf("\nPlaying song 3\n");
                    song = 3;
                    return;
                }
                else if (song == 2) {
                    uLCD.printf("\nPlaying song 1\n");
                    song = 1;
                    return;
                }
                else if (song == 3) {
                    uLCD.printf("\nPlaying song 2\n");
                    song = 2;
                    return;
                }
            }
            else if (mode == 2) { //CHOOSE MODE
                song = 1;
                uLCD.cls();
                uLCD.color(WHITE);
                uLCD.printf("\nSelect Song\n");
                uLCD.color(GREEN);
                uLCD.printf("Song 1\n");
                uLCD.printf("Song 2\n");
                uLCD.printf("Song 3\n");
                while (true) {
                    if (!sw3) {
                        uLCD.cls();
                        uLCD.color(GREEN);
                        uLCD.printf("\nPlaying song %d\n", song);
                        play = 1;
                        return;
                    }
                    else if (output == 1) {
                    uLCD.cls();
                    uLCD.color(WHITE);
                    uLCD.printf("\nSelect Song\n");
                    uLCD.color(RED);
                    uLCD.printf("Song 1\n");
                    uLCD.color(GREEN);
                    uLCD.printf("Song 2\n");
                    uLCD.printf("Song 3\n");
                    song = 1;
                    wait(0.1);
                    }
                    else if (output == 2) {
                        uLCD.cls();
                        uLCD.color(WHITE);
                        uLCD.printf("\nSelect Song\n");
                        uLCD.color(GREEN);
                        uLCD.printf("Song 1\n");
                        uLCD.color(RED);
                        uLCD.printf("Song 2\n");
                        uLCD.color(GREEN);
                        uLCD.printf("Song 3\n");
                        song = 2;
                        wait(0.1);
                    }
                    else if (output == 3) {
                        uLCD.cls();
                        uLCD.color(WHITE);
                        uLCD.printf("\nSelect Song\n");
                        uLCD.color(GREEN);
                        uLCD.printf("Song 1\n");
                        uLCD.printf("Song 2\n");
                        uLCD.color(RED);
                        uLCD.printf("Song 3\n");
                        song = 3;
                        wait(0.1);
                    }
                }
                return;
            }
            else if (mode == 3) { //GAME
                song = 1;
                play = 1;
                int x = 0;
                int y = 64;
                int score = 0;
                int match = 0;       
                int color = 0;
                uLCD.background_color(BLACK);
                uLCD.cls();
                uLCD.circle(64, 64, 10, WHITE);
                while(sw2) {
                    if (color%2 == 1) {
                        uLCD.filled_circle(x, y, 5, RED);
                    }
                    else {
                        uLCD.filled_circle(x, y, 5, BLUE);
                    }
                    if ((x > 64 && x < 104) && output == 1 && !match && color%2) {
                        score += 2;
                        match = 1;
                    }
                    else if ((x > 64 && x < 104) && output == 1 && !match && !(color%2)) {
                        score += 1;
                        match = 1;
                    }
                    if (x > 138) {
                        uLCD.circle(64, 64, 10, WHITE);
                        match = 0;
                        color ++;
                        x = 0;
                    }
                    uLCD.filled_circle(x, y, 5, BLACK);
                    uLCD.circle(64, 64, 10, WHITE);
                    uLCD.locate(1, 2);
                    uLCD.printf("\nSCORE : %d\n", score);
                    if (color%2 == 1){
                        x = x + 10;
                    }
                    else{
                        x = x + 5;
                    }  
                }
                uLCD.cls();             
                return;
            }  
        }
        
    }
}

void loadSignal(void) {
    int signalLength = 98;
    int bufferLength = 32;
    float signal[signalLength];
    char serialInBuffer[bufferLength];
    int i = 0;
    int serialCount = 0;
    while(i < signalLength) {
        if(pc.readable()) {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;
            if(serialCount == 3) {
                serialInBuffer[serialCount] = '\0';
                signal[i] = atoi(serialInBuffer);
                serialCount = 0;
                i++;
            }
        }
    }
    for (i = 0; i < 42; i++) {
        song1[i] = signal[i];
    }
    for (i = 42; i < 66; i++) {
        song2[i-42] = signal[i];
    }
    for (i = 66; i < 98; i++) {
        song3[i - 66] = signal[i];
    }
    wait(1.0);   
}



int main() {  
    t1.start(DNN);
    t2.start(PlaySong);
    while(1) {
        if (!sw2)
        SelectMode();
    };
    t2.start(loadSignal);  
}