import random



with open("/X-E-Speech/filelist/vctk_audio_phone.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()


total_lines = len(lines)
num_lines_file1 = int(total_lines * 0.9)
num_lines_file2 = total_lines - num_lines_file1


random_lines = random.sample(lines, k=total_lines)


lines_file1 = sorted(random_lines[:num_lines_file1])
lines_file2 = sorted(random_lines[num_lines_file1:])

with open("/X-E-Speech/filelist/vctk_audio_phone-train.txt", "w", encoding="utf-8") as f:
    f.writelines(lines_file1)

with open("/X-E-Speech/filelist/vctk_audio_phone-test.txt", "w", encoding="utf-8") as f:
    f.writelines(lines_file2)