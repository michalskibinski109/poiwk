# Server for video processing

# usage 
#### clone repository to your local machine
```bash
git clone https://github.com/michalskibinski109/poiwk
```


#### install dependencies
```bash
pip install -r requirements.txt
```
#### run server
```bash
python main.py
```
#### open browser and go to
```bash
http://localhost:5000
```
#### then choose file from `videos` folder and click `upload`

### to use your own model 
1. replace `model.h5` file 
2. check input shape of your model and change `input_shape` in `VideoRunner` class
3. change `__prepare_images` function in `VideoRunner` class to prepare images for your model. Image should be processed in exactly the same way as it was during training.

If you have any questions, feel free to contact me: 

[itaknieodpowiem@spierdalaj.pwr]()