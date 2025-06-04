from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
while True:
    a=input('enter:-')
   #[0,180,0,180,180]
    #kit.servo[5].angle = 0
    kit.servo[1].angle = 180
    kit.servo[2].angle = 0
    kit.servo[3].angle = 180
    kit.servo[4].angle = 180
    
    a=input('enter:-')
    #kit.servo[5].angle = 180
    kit.servo[1].angle = 180
    kit.servo[2].angle = 180
    kit.servo[3].angle = 0
    kit.servo[4].angle = 0
	

 
