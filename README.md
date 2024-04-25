# Animal Sound Showdown

* How to reproduce:

1. Go to `Model.ipynb` and run every cell in Colab

   * Notice: the dataset is in the GitHub Home named `Data.zip`, you can customize your own path and put it on. Don't forget to change the corresponding code.

2. Download model and features

   * You can also get them in the GitHub Home

3. Set your Raspberry Pi, put model and features inside, and use flask to run `app.py`

   * Use virtual environment and download these libraries 

   * ```
     flask,skimage,sklearn,ensorflow,librosa,numpy,tflite_runtime,traceback
     ```

4. Download the Flutter app on your phone from `app` folder
   * If it is needed, you need to change the host of URL in the `main.dart`

