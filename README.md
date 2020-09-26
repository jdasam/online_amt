Real-time Automatic Piano Transcription System
=======

This is the code for our Real-time Automatic Piano Transcription System, which was presented in SK Telecom Tech Gallery in Pangyo, Korea.
The documentation is currently on working.

Requirements
------
- python_rtmidi==1.1.2
- Flask==1.1.2
- scipy==1.4.1
- numpy==1.16.2
- PyAudio==0.2.11
- librosa==0.7.2
- matplotlib==3.1.1
- torch==1.6.0
- rtmidi==2.3.4


Usage
-----
#### With a web browser visualization

```$ python run_on_web.py ```

Then, open http://127.0.0.1:5000/ with your browser.
After the page is opened, the AMT model will automatically run until a keyboard interrupt.

#### With a matplotlib visualization
```$ python run_on_plt.py ```
