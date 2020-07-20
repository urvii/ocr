from demo import make_animation
from skimage import img_as_ubyte





# try:
#     from PIL import Image
# except ImportError:
#     import Image
# import pytesseract


from demo import load_checkpoints


def deepfake_m(source_image,driving_video):
    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                            checkpoint_path='/content/gdrive/My Drive/first-order-motion-model/vox-cpk.pth.tar')
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

    #save resulting video
    imageio.mimsave('../generated.mp4', [img_as_ubyte(frame) for frame in predictions])
    #video can be downloaded from /content folder

    # HTML(display(source_image, driving_video, predictions).to_html5_video())

    return predictions