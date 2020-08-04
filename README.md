# Template-Matching
Computer Vision Template Matching Using OpenCV Python

#### Install required packages using pip

```python
pip install -r requiremennt.txt
```

#### Run with default parameter
```python
python Template_Matching.py
```

#### Default Argument List
* match_threshold = 0.5
* Non-Max-Supression Threshold = 0.5
* input_image = ./data/Face.jpg
* template_directory = ./templates
* output_directory = ./output


#### Run with external argument
```python
python Template_Matching.py -match_threshold 0.7 -nmst 0.4 -input_image ./data/Face.jpg -template_directory ./templates -output ./output
```
