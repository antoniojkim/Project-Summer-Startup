# Project-Summer-Startup

Summary of Ideas
==================
1) Firefighting Drone
2) Autonomous Vehicles
3) Image Generation
4) Stock market predictor ( very far fetched but yolo)

Firefighting Drone Detailed Idea
------------------
* Essentially, the drone is dispatched in a closed location, and identifies key features for firefighter.
  * Detects building's structural weaknesses
  * Locations of victims
  * Potential Hazards
* Locates safest path to any victims
* Transmits data in real time so algorithm can adjust for any changes
* Requirements:
  * Drone has to be able to survive fire
  * Drone has to be able to survive falling debris
* Components:
  * Raspberry pi on drone
  * LIDAR on drone
  * If possible, sonar detectors, like small arduino ones, that are used for drone boundar detection
  * Gyroscope
  * Altimeter

Image Generation
--------------
* Train a neural network on a set of images, distinct, but sharing some aspect, an example would be a set of different faces
* Then have it create that that object that it trained, so in the case of a face, it would draw out a face
* Should also be able to create elements of the object, so in case of a face, it could draw just the eye 

Algorithms/Designs
---------------
* So we need one for obstacle detection
* Image detection and classification
* Converting LIDAR data into useful input
* Calculating saftest path/shortest path (depending on application)
* Should be very efficient
  * one way to achieve this is to send data to a nearby server and have it calculate whatever
* Drone should be as light and aerodynamically efficient as possible 
* In case of large areas of data collection, we dispatch multiple drones:
  * Area first gets split up into chunks
  * Each drone is dispatched to a different area
  * Each drone can either do a different task, or the same one, and then data is all sent to the nearby server
* Once data is collected and sent to server, it should be forgotten by the drone, and just updated in real time
  * Ideally in C to destroy memory
 



Resources for machine learning
-------------------
* http://www.projectforrest.com/path/83
* http://www.deeplearningbook.org/
* http://neuralnetworksanddeeplearning.com/chap1.html
