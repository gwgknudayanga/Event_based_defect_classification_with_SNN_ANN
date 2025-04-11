* This code is related to our conference paper **"Event-Based Classification of Defects in Civil Infrastructures with Artificial and Spiking Neural Networks"** published at 
https://link.springer.com/chapter/10.1007/978-3-031-43078-7_51. If you use our code or refer to our paper, please cite this paper !!!


* We experimented both image-based and event-based crack and spalling defect classification with ANNs and SNNs.
* We use selected SOTA image-based datasets. To generate the event-based equivalents from those images , we simulated camera projection for a given image and generate a video and then that video is fed to v2e event camera simulator to generate events. (generate_v2e_events.py)

* When feeding event-based data to ANNs, we used voxel-grid encoding.
* When feeding event-based data to SNNs, we used event-cube encoding as we feed direct events.
* When feeding image-based data to SNNs, we used a poisson generator to generate rate coded direct events.
