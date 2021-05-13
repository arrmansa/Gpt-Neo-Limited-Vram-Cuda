# Gpt-Neo-Limited-Vram-Cuda
 A notebook that runs GPT-Neo with low vram (8 gb or 6 gb) and cuda acceleration by loading it into gpu memory in smaller parts.<br>
## Why?
This method may perhaps provide a much more significant acceleration if used with much larger models and higher vram (and ram-vram bandwidth) gpus, making it possible to run large models (such as gpt-3) on high end consumer hardware.
### Notes
The lack of any significant speed difference between splitting the model into 32 parts (105 second runtime) and 2 parts (85 second runtime) indicates that ram->vram transfer is the major bottleneck for this process. <br> 
There may be a possibility of transferring blocks to gpu with multiprocessing, since the cpu load also seems single core, and having tensors transferred to gpu serially is not important. <br>
The Pytorch only supporting inplace transfers of modules from gpu to cpu made this more complex than is should be.
