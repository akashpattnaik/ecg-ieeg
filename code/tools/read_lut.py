import re
import pandas as pd

def read_lut(lut_path="/gdrive/public/USERS/pattnaik/VEP_atlas_shared/data/VepFreeSurferColorLut.txt"):
    """
    from https://gist.github.com/rmukh/a655b5ac16a37d72cb6f20747664e4af
    The link to the table: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    It is also available locally if FreeSurfer is installed: $FREESURFER_HOME/FreeSurferColorLUT.txt
    Read and store FreeSufer LUT color table
    The result is two variables:
    rgb - numpy array where 0th dimention is the region code (first column) and 3 other 
    dimensions are for RBG values. Final fimension: (number of regions x 3)
    label_names - dictinory where key = region code, and value = label name
    """

    # replace the path to the FreeSurferColorLUT.txt with your own
    # in linux you can do: echo $FREESURFER_HOME/FreeSurferColorLUT.txt

    with open(lut_path, 'rb') as file:
        raw_lut = file.readlines()

    # read and process line by line
    rows = []
    pattern = re.compile(r'\d{1,5}[ ]+[a-zA-Z-_0-9*.]+[ ]+\d{1,3}[ ]+\d{1,3}[ ]+\d{1,3}[ ]+\d{1,3}')
    for line in raw_lut:
        decoded_line = line.decode("utf-8")
        if pattern.match(decoded_line):
            s = decoded_line.rstrip().split(' ')
            s = list(filter(None, s))
            s[0] = int(s[0])
            rows.append(s[0:2])

    df = pd.DataFrame(rows)
    df.columns = ['index', 'label']
    return df.set_index('index',drop=True)

