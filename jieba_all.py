# encoding=utf-8
import jieba

jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持



from utils import load_filepaths_and_text

filelist="dataset/ESD_16k/output_cn.txt"

print("START:", filelist)
filepaths_and_text = load_filepaths_and_text(filelist)
jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
for i in range(len(filepaths_and_text)):
    original_text = filepaths_and_text[i][1]
    
    seg_list = jieba.cut(original_text, cut_all=False,use_paddle=True)
    cleaned_text="-".join(seg_list)  # 精确模式
    filepaths_and_text[i][1] = cleaned_text

new_filelist = filelist + "." + 'jieba.txt'
with open(new_filelist, "w", encoding="utf-8") as f:
    f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])