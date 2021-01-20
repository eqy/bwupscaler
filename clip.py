import ast
import csv
import ffmpeg
import os
import shutil
import subprocess


class Clip(object):
    def __init__(self, filename, crop=None, exclude_segments=None):
        # TODO: handle crop and exclude segments if added later
        self.filename = filename
        self.crop = crop
        self.exclude_segments = exclude_segments
        probe = ffmpeg.probe(filename)

        for meta in probe['streams']:
            if meta['codec_type'] == 'video': 
                video_meta = meta
                break
        # get metadata for video clip
        self.height = int(video_meta['height'])
        self.width = int(video_meta['width'])
        # LOL THIS LOOKS UNSAFE
        self.framerate = eval(video_meta['avg_frame_rate'])


    def extract_data(self, dest_path, firstval=False):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        traindir = os.path.join(dest_path, 'train')
        valdir = os.path.join(dest_path, 'val')
        if not os.path.exists(traindir):
            os.makedirs(traindir)
        if not os.path.exists(valdir):
            os.makedirs(valdir)

        basename = os.path.splitext(os.path.basename(self.filename))[0]
        fps_str = f'fps={str(int(self.framerate))}'
        jpeg_str = os.path.join(dest_path, f'{basename}_%d.jpg')
        ffmpeg_cmd = ['ffmpeg', '-i', self.filename, '-q:v', '5', '-vf', fps_str, jpeg_str]
        subprocess.call(ffmpeg_cmd)

        count = 0
        jpgs = list()
        for dirpath, dirnames, filenames in os.walk(dest_path):
            if dirpath == traindir or dirpath == valdir:
                continue
            for filename in filenames:
                name, ext = os.path.splitext(filename)
                if ext == '.jpg':
                    count += 1
                    num = int(name.split('_')[-1])
                    jpgs.append((filename, num))
        half = count/2
        for jpg in jpgs:
            name = jpg[0]
            num = jpg[1]
            if firstval:
                if num < half:
                    dest = valdir
                else:
                    dest = traindir
            else:
                if num < half:
                    dest = traindir
                else:
                    dest = valdir
            shutil.move(os.path.join(dest_path, name), dest)


def load(filename):
    clips = list()
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            clip_filename = row[0]
            crop = None
            exclude_segments = None
            if len(row) > 1:
                crop = ast.literal_eval(row[1])
            if len(row) > 2:
                exclude_segments = ast.literal_eval(row[2])
            clip = Clip(clip_filename, crop, exclude_segments)
            clips.append(clip)
    return clips

def main():
    clips = load('clips.csv')
    for i, clip in enumerate(clips):
        clip.extract_data('data5', i % 2 == 1)

if __name__ == '__main__':
    main()
