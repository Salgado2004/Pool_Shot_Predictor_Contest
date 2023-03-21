![image](https://user-images.githubusercontent.com/53799801/224310584-ed00aefc-dd5e-4549-ad7f-b552d2f6b787.png)

<p align="justify">Programming contest hosted by CV Zone, sponsored by <a href="https://www.computervision.zone/?ns_url=1Zy&mid=9916343"><b>NVIDIA</b></a>. Where the objective is to <b>predict the paths of the ball on the pool table and whether they will go in the hole</b>. </p>

<p align="justify">The main method used is color and shape detection to discern the objects on the pool table. To each object, the program defines a different HSV filter. With the coordinates of each object, it calculates the likely path of the ball based on linear algebra. As the program can't estimate the exact location of the cue, thus creating many different possible outcomes, it also calculates which is the most likely path. The most likely path is shown during the shot.</p>

<p align="justify">It has been defined labels in front of the objects, to help me understand what the program was detecting during the development. Later on, I decided to keep them just for style.</p>

<b>** If you like this project, please vote for me in <a href="https://www.computervision.zone/pool-shot-predictor/">https://www.computervision.zone/pool-shot-predictor/</a></b> 

<h2>Result samples</h2>
<img src="https://user-images.githubusercontent.com/53799801/226636165-55f0aef3-b3c6-404c-b421-a3ddb2233ac7.png"><br>
<img src="https://user-images.githubusercontent.com/53799801/226637523-9b5d2650-454e-4213-adf7-c215971ab9e4.png"><br>


<b>** If you like this project, please vote for me in <a href="https://www.computervision.zone/pool-shot-predictor/">https://www.computervision.zone/pool-shot-predictor/</a></b> 

<h2>&copy; Leonardo Salgado</h2>
<img src="https://user-images.githubusercontent.com/53799801/224312217-dd02b5af-a6b4-45e6-b1df-db3f0b9dc702.png"><br>



