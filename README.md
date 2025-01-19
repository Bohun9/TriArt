## TriArt
TriArt is a painter that draws images using triangles as the basic shapes.
It uses an evolutionary process to develop progressively better generations of images.

![Mona Lisa](examples/mona_lisa.gif)

### Dependencies

```bash
pip install -r requirements.txt
sudo apt install libopencv-dev python3-opencv
```

While OpenCV can also be installed via pip, using the system package manager is recommended to avoid potential
[conflicts](https://stackoverflow.com/questions/46449850/how-to-fix-the-error-qobjectmovetothread-in-opencv-in-python).

### Usage

This command will start the evolution and save the results and statistics to the `saved/<save_name>.pkl` file:
```bash
python3 -m src.triart <image_path> --save_name <save_name> [options]
```

For a full list of avaiable options, use:
```bash
python3 -m src.triart -h
```

To generate frames of the best individuals, use:
```bash
python3 -m src.generate_visuals <image_path> <save_name>
```
