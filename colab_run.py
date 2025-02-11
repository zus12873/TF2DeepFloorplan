import tensorflow as tf
import sys
from dfp.net import *
from dfp.data import *
import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
from argparse import Namespace
import os
import gc
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from dfp.utils.rgb_ind_convertor import *
from dfp.utils.util import *
from dfp.utils.legend import *
from dfp.utils.settings import *
from dfp.deploy import *

# 打印 GPU 是否可用（可选择注释掉，不需要输出）
# print(tf.test.is_gpu_available())
# print(tf.config.list_physical_devices('GPU'))

img_path = './TF2DeepFloorplan/resources/30939153.jpg'
inp = mpimg.imread(img_path)

args = parse_args("--tomlfile ./TF2DeepFloorplan/docs/notebook.toml".split())
args = overwrite_args_with_toml(args)
args.image = img_path
result = main(args)

# 保存原始图像和结果图像
plt.subplot(1, 2, 1)
plt.imshow(inp); plt.xticks([]); plt.yticks([]);
plt.subplot(1, 2, 2)
plt.imshow(result); plt.xticks([]); plt.yticks([]);
plt.savefig('original_and_result_image.png')

model, img, shp = init(args)
logits_cw, logits_r = predict(model, img, shp)
logits_r = tf.image.resize(logits_r, shp[:2])
logits_cw = tf.image.resize(logits_cw, shp[:2])

r = convert_one_hot_to_image(logits_r)[0].numpy()
cw = convert_one_hot_to_image(logits_cw)[0].numpy()

# 保存 r 和 cw 的图像
plt.subplot(1, 2, 1)
plt.imshow(r.squeeze()); plt.xticks([]); plt.yticks([]);
plt.subplot(1, 2, 2)
plt.imshow(cw.squeeze()); plt.xticks([]); plt.yticks([]);
plt.savefig('r_and_cw_image.png')

r_color, cw_color = colorize(r.squeeze(), cw.squeeze())

# 保存着色后的 r 和 cw 图像
plt.subplot(1, 2, 1)
plt.imshow(r_color); plt.xticks([]); plt.yticks([]);
plt.subplot(1, 2, 2)
plt.imshow(cw_color); plt.xticks([]); plt.yticks([]);
plt.savefig('r_and_cw_colored.png')

newr, newcw = post_process(r, cw, shp)

# 保存后处理的 newr 和 newcw 图像
plt.subplot(1, 2, 1)
plt.imshow(newr.squeeze()); plt.xticks([]); plt.yticks([]);
plt.subplot(1, 2, 2)
plt.imshow(newcw.squeeze()); plt.xticks([]); plt.yticks([]);
plt.savefig('newr_and_newcw.png')

newr_color, newcw_color = colorize(newr.squeeze(), newcw.squeeze())

# 保存后处理后着色的图像
plt.subplot(1, 2, 1)
plt.imshow(newr_color); plt.xticks([]); plt.yticks([]);
plt.subplot(1, 2, 2)
plt.imshow(newcw_color); plt.xticks([]); plt.yticks([]);
plt.savefig('newr_and_newcw_colored.png')

# 合并后的图像保存
plt.imshow(newr_color + newcw_color); plt.xticks([]); plt.yticks([]);
plt.savefig('final_merged_image.png')

# 添加图例并保存
over255 = lambda x: [p / 255 for p in x]
colors2 = [over255(rgb) for rgb in list(floorplan_fuse_map.values())]
colors = ["background", "closet", "bathroom",
          "living room\nkitchen\ndining room",
          "bedroom", "hall", "balcony", "not used", "not used",
          "door/window", "wall"]
f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
handles = [f("s", colors2[i]) for i in range(len(colors))]
labels = colors
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)

fig = legend.figure
fig.canvas.draw()

# 保存图例
plt.xticks([]); plt.yticks([]);
plt.savefig('legend_image.png')

# 清理缓存，释放内存
gc.collect()
