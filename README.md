# Gpt-Neo-Limited-Vram-Cuda
 A notebook that runs GPT-Neo with low vram (8 gb or 6 gb) and cuda acceleration by loading it into gpu memory in smaller parts.<br>
## Why?
This method may perhaps provide a much more significant acceleration if used with much larger models and higher vram (and ram-vram bandwidth) gpus, making it possible to run large models (such as gpt-3) on high end consumer hardware.
### Notes
1. The lack of any significant speed difference between splitting the model into 32 parts (105 second runtime) and 2 parts (85 second runtime) indicates that ram->vram transfer is the major bottleneck for this process. <br> 
2. There may be a possibility of transferring blocks to gpu with multiprocessing, since the cpu load also seems single core, and having tensors transferred to gpu serially is not important. <br>
3. Gpu bandwith tested using debug cell at the end shows a rate of 4 GB/s (approx 490gb in 120 second) which is 10% speed of my dual channel 2666Mhz DDR4 ram max theoretical bandwidth (40 GB/s) and 25% of the pci express x16 gen 3 (16 GB/s) bandwidth.<br>
4. The Pytorch only supporting inplace transfers of modules from gpu to cpu made this more complex than is should be.
