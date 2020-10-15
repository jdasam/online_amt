Real-time Automatic Piano Transcription System
=======

![Screen shot](https://github.com/jdasam/jdasam.github.io/blob/gh-pages/assets/images/online_amt.png?raw=true)
This is the code for our Real-time Automatic Piano Transcription System, which was presented in SK Telecom Tech Gallery in Pangyo, Korea.
The documentation is currently on working.

The system is based on the AMT model based on [Polyphonic Piano Transcription Using Autoregressive Multi-State Note Model
 (ISMIR 2020)](https://program.ismir2020.net/poster_3-17.html). For the detailed explanation on the system implementation, please refer [ISMIR 2020 LBD](https://program.ismir2020.net/lbd_444.html)

Requirements
------
- Flask==1.1.2
- scipy==1.4.1
- numpy==1.16.2
- PyAudio==0.2.11
- librosa==0.7.2
- matplotlib==3.1.1
- torch==1.6.0
- rtmidi==2.3.4
- python_rtmidi==1.1.2
- numba==0.48

Pre-trained Model
-----
The pre-trained model for AMT is uploaded with Git-LFS. If you are not familiar with Git-LFS, you can download it from [here](https://drive.google.com/file/d/12DnYJJ6YKpsoEkXI9fYUTTd4wOeY_rlB/view?usp=sharing).

The model was trained with [MAESTRO v.2.0.0](https://magenta.tensorflow.org/datasets/maestro) based on the code by [Jongwook Kim](https://github.com/jongwook/onsets-and-frames)


Usage
-----
#### Caution
If you run the code on a laptop while using a laptop microphone input, the fan noise of laptop will cause severe degradation of AMT performance. We recommend you to use an external microphone, or internal audio such as Soundflower.

#### With a web browser visualization

```$ python run_on_web.py ```

Then, open http://127.0.0.1:5000/ with your browser.
After the page is opened, the AMT model will automatically run until a keyboard interrupt.


#### With a matplotlib visualization
```$ python run_on_plt.py ```



