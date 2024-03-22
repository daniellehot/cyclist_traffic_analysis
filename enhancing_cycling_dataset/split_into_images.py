import os, shutil
import cv2

def main():
    videos = [f for f in os.listdir(".") if f.endswith(".MP4")]

    for v in videos:
        # get file path for desired video and where to save frames locally
        path_to_save = os.path.join(os.path.join(".", v.replace(".MP4", "")))
        if os.path.exists(path_to_save):
            shutil.rmtree(path_to_save)
        os.makedirs(path_to_save)

        cap = cv2.VideoCapture(v)
        if (cap.isOpened() == False):
            print('Cap is not open')

        current_frame = 0 
        # cap opened successfully
        while(cap.isOpened()):

            # capture each frame
            ret, frame = cap.read()
            if(ret == True):
                if current_frame % 30 == 0:
                    # Save frame as a jpg file
                    file_name = str(current_frame).zfill(6) + '.png'
                    print(f'Saving {os.path.join(path_to_save, file_name)}')
                    cv2.imwrite(os.path.join(path_to_save, file_name), frame)
                # keep track of how many images you end up with
                current_frame += 1
                print(f"Video {v} Frame {current_frame}")
            else:
                break
            
            if current_frame == 1800:
                break
        # release capture 
        cap.release()
        print('done')

if __name__ == '__main__':
    main()