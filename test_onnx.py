import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import glob
import random

def postprocess(img):
    img = (img + 1.0) * 127.5 + 0.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = img[0].transpose((1, 2, 0))
    return img

def inference(path_list, img_path, mask_path):
    encoder_sess = ort.InferenceSession(path_list[0], providers = ("CPUExecutionProvider",))
    generator_sess = ort.InferenceSession(path_list[1], providers = ("CPUExecutionProvider",))
    mappingnet_sess = ort.InferenceSession(path_list[2], providers = ("CPUExecutionProvider",))

    img = np.array(Image.open(img_path).resize((256, 256),Image.Resampling.BILINEAR))
    mask = np.array(Image.open(mask_path).resize((256, 256), Image.Resampling.NEAREST))
    Image.fromarray(img).save("onnx_truth.png")

    mask = mask / 255.0
    mask = 1 - mask
    mask = mask.transpose((2,0,1))
    img = img.transpose((2, 0, 1))
    mask = np.expand_dims(mask, axis=0)
    img = np.expand_dims(img, axis=0)
    img = (img / 127.5) - 1.0
    masked_img = img * mask
    mask = mask[:,0:1,:,:]
    input_x = np.concatenate((masked_img, mask), axis=1).astype(np.float32)
    noise = np.random.randn(1, 512).astype(np.float32)

    ws = mappingnet_sess.run(None, {"noise": noise})[0]
    encoder_out = encoder_sess.run(None, {'input': input_x, "in_ws": ws})
    # ws = np.expand_dims(encoder_out[0], axis=1)
    input_dict = {f"en_feats{i - 1}": encoder_out[i] for i in range(1, len(encoder_out))}
    # ws = np.repeat(ws, 11, axis=1)
    input_dict["style"] = encoder_out[0]
    gen_out = generator_sess.run(None, input_dict)[0]
    comp_img = masked_img + gen_out * (1 - mask)
    masked_img = masked_img + (1 - mask)
    masked_img = postprocess(masked_img)
    comp_img = postprocess(comp_img)
    Image.fromarray(masked_img).save("onnx_masked.png")
    Image.fromarray(comp_img).save("onnx_out.png")
    print()


if __name__ == '__main__':
    encoder_path = "./encoder.onnx"
    generator_path = "./generator.onnx"
    mappingnet_path = "./mapping.onnx"
    model_paths = [encoder_path, generator_path, mappingnet_path]
    img_path = "/home/codeoops/CV/inpainting_baseline/example/places/Places/thick_512/imgs"
    mask_path = "/home/codeoops/CV/inpainting_baseline/example/places/Places/thick_512/masks"

    imgs = sorted(glob.glob(img_path + "/*.jpg") + glob.glob(img_path + "/*.png"))
    masks = sorted(glob.glob(mask_path + "/*.jpg") + glob.glob(mask_path + "/*.png"))

    rand_id = random.randint(0, len(imgs) - 1)
    inference(model_paths, imgs[rand_id], masks[rand_id])


