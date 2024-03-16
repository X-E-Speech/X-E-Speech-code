#encoding:utf-8

#JVSデータセットに対し、各音声ファイルに対しファイルパス、話者id、発話内容の3つを1行ずつにまとめ、学習用と推論用の2種類のtxtファイルに出力するスクリプト
#各話者についてnonpara30とparallel100内のwavファイルを対象とする
#また、音声ファイルのサンプリングレートを22050Hzに変換する

import sys
import os
import re
import pyopenjtalk
import librosa
import soundfile as sf
import random

#JVSコーパスのデータセットへのパス
jvs_dataset_path = "/jvs_ver1/jvs_ver1"
#txtファイル、サンプリングレート変換後のwavファイルの出力用ディレクトリ
output_dir = "/jvs_ver1/jvs_preprocessed/"
#出力するtxtファイル(学習用データセット用)の名前
output_train_txtfile_path = os.path.join(output_dir, "jvs_preprocessed_for_train_nodouhao_nokongge.txt")
#出力するtxtファイル(推論用データセット用)の名前
output_validation_txtfile_path = os.path.join(output_dir, "jvs_preprocessed_for_validation_nodouhao_nokongge.txt")
#推論用データとして用いる音声ファイルの数
validation_file_number = 10
#出力する音声ファイルの出力先ディレクトリ
output_wav_dir = os.path.join(output_dir, "jvs_wav_preprocessed")
#音声ファイルを書き出す際のサンプリングレート
output_wav_sampling_rate = 16000
#前処理の対象とするwavファイルの最小の長さ　これより短いwavファイルはtxtファイルへの出力の対象としない
min_wav_length = 16000*0.5

#後述の処理により(wavファイルへのパス, 話者id, 発話内容)を保持するlist　空のlistで初期化しておく
wavfilepath_speakerid_text = []

#乱数のシードを設定
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)

#transcripts_utf8.txtを元にwavファイルへのパス, 話者, 発話内容の3つを関連づける関数
def preprocess_using_transcripts(speaker_id, speaker_id_on_jvs, filedir):
	#"transcripts_utf8.txt"へのパス
	transcripts_utf8_path = os.path.join(filedir, "transcripts_utf8.txt")
	#ディレクトリ"wav24kHz16bit"へのパス
	wav24kHz16bit_dir = os.path.join(filedir, "wav24kHz16bit")
	#transcripts_utf8.txtを開き1行づつ処理。1行につき1つのwavファイル、発話が関連づけられている
	with open(transcripts_utf8_path) as f:
		for line in f:
			line = line.split(":")
			wav_file_name = f"{line[0]}.wav"#読み込むwavファイルの名前
			load_wav_file_path = os.path.join(wav24kHz16bit_dir, wav_file_name)#読み込むwavファイルへのパス
			output_wav_path = os.path.join(output_wav_dir, speaker_id_on_jvs, wav_file_name)#出力する音声ファイルの出力パス
			#wavファイルが存在する場合以下を実行
			if(os.path.exists(load_wav_file_path)):
				# if not os.path.exists(output_wav_path):
				# 	print(output_wav_path)
				# 	##########wavファイルに関する処理##########
				# 	##wavファイルをサンプリングレートoutput_wav_sampling_rate[Hz]に変換して保存する
				# 	loaded_wav_file, _ = librosa.core.load(load_wav_file_path, sr=output_wav_sampling_rate, mono=True)#wavファイルをサンプリングレートoutput_wav_sampling_rate[Hz]に変換して読み込み
				# 	if(loaded_wav_file.shape[0]<min_wav_length):#wavファイルが閾値よりも短いならば前処理の対象から除外
				# 		print(f"excluded from preprocessing:{wav_file_name} (len:{loaded_wav_file.shape[0]})")
				# 		continue
					
				# 	os.makedirs(os.path.join(output_wav_dir, speaker_id_on_jvs), exist_ok=True)#出力先ディレクトリがなければ作成
				# 	sf.write(output_wav_path, loaded_wav_file, samplerate=output_wav_sampling_rate, subtype="PCM_16")#16[bit]でwavファイルを保存

				##########textに関する処理##########
				##読み込んだ発話内容を音素へと変換する
				text = line[1]#wavファイルに対応する発話内容を取得
				text_converted = text.strip()#改行コードを削除
				text_converted = re.sub('・|・|「|」|』', '', text_converted)#発音とは無関係な記号を削除
				text_converted = re.split('、|,|，|。|『', text_converted)#句読点、もしくは『で分割
				#分割した各文字列について音素列への変換を実行
				text_converted = [pyopenjtalk.g2p(element) for element in text_converted if(not element=="")]
				
				#分割した各文字列についてスペースをカンマに変換
				#text_converted = [element.replace(" ",",") for element in text_converted]
				#各発話(音素列)をスペース区切りで接合
				text_converted = ','.join(text_converted)
				text_converted = re.sub(r'\s', '', text_converted)#発音とは無関係な記号を削除
				#文字列にpauが含まれている(解釈に失敗した記号)が含まれていれば処理を飛ばす　
				if("pau" in text_converted):
					print(f"\"pau\" is included:{text_converted}")
					continue
				#(wavファイルへのパス, 話者id, 発話内容)をlistへ追加
				wavfilepath_speakerid_text.append((output_wav_path, text_converted))

#jvs001~jvs100まで順に見ていく
for speaker_id in range(0,100):
	#対象の音声ファイル群の入ったディレクトリ
	speaker_id_on_jvs = f"jvs{speaker_id+1:03}"
	#対象の音声ファイル群の入ったディレクトリへのパス
	speaker_dir = os.path.join(jvs_dataset_path, speaker_id_on_jvs)
	#nonpara30とparallel100に関して、各音声ファイルに対しファイルパス、話者id、発話内容の3つを1行ずつにまとめ関連づける
	preprocess_using_transcripts(
		speaker_id=speaker_id,
		speaker_id_on_jvs=speaker_id_on_jvs,
		filedir=os.path.join(speaker_dir, "nonpara30")
	)
	preprocess_using_transcripts(
		speaker_id=speaker_id,
		speaker_id_on_jvs=speaker_id_on_jvs,
		filedir=os.path.join(speaker_dir, "parallel100")
	)
	print(speaker_id_on_jvs, "preprocessed")

#出力用ディレクトリがなければ作成
os.makedirs(output_dir, exist_ok=True)

####各音声ファイルに対しファイルパス、話者id、発話内容の3つを1行ずつにまとめたtxtファイルを出力####

#list"wavfilepath_speakerid_text"のうち、どのindexのものをvalidation用データとするかをランダムに決める
#1つづつindexを生成、すでに生成されたindexと被っていなければvalidation_file_indexに追加
validation_file_index = []
while(len(validation_file_index)<validation_file_number):
	index = random.randrange(0, len(wavfilepath_speakerid_text))
	if(not (index in validation_file_index)):
		validation_file_index.append(index)

#学習用、推論用データセットについてまとめたtxtファイルを出力
with open(output_train_txtfile_path, 'w') as train_file:
	with open(output_validation_txtfile_path, 'w') as validation_file:
		for index, (wavfilepath, text) in enumerate(wavfilepath_speakerid_text, 0):
			if(index in validation_file_index):
				validation_file.write(f"{wavfilepath}|{text}\n")
			else:
				train_file.write(f"{wavfilepath}|{text}\n")

print(f"train dataset size : {len(wavfilepath_speakerid_text) - validation_file_number}")
print(f"validation dataset size : {validation_file_number}")


