import numpy as np
import serial
import time


waitTime = 0.1
signalLength = 98
# Generate signal table
signalTable = [
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261,
    392, 392, 349, 349, 330, 330, 294,
    392, 392, 349, 349, 330, 330, 294,
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261,      #song1
    392, 330, 330, 349, 294, 294,
    261, 294, 330, 349, 392, 392, 392,
    392, 330, 330, 349, 294, 294, 
    261, 330, 392, 392, 261,                #song2
    261, 294, 330, 261, 261, 294, 330, 261,
    330, 349, 392, 330, 349, 392,
    392, 440, 392, 349, 330, 261,
    392, 440, 392, 349, 330, 261,
    294, 392, 261, 294, 392, 261            #song3   
]
# output formatter
formatter = lambda x: "%d" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
for data in signalTable:
    s.write(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)
s.close()

print("Signal sended")