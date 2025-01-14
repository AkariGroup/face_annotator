# face_annotator

OAK-Dのtracking機能を使い、認識できなかったフレームを自動でYOLOアノテーションし、保存するアプリ

## セットアップ
1. submoduleの更新  
`git submodule update --init`  

1. 仮想環境の作成  
`python3 -m venv venv`  
`. venv/bin/activate`  
`pip install -r requirements.txt`  

## 実行
`python3 face_annotator.py -n "名前" -i "id"`  

オプションは下記  
- `-f`, `--fps`: カメラ画像の取得PFS。デフォルトは5。このfpsで画像を保存していく。OAK-Dの性質上、推論の処理速度を上回る入力を与えるとアプリが異常終了しやすくなるため注意。
- `-n`. `--name`: 保存先のディレクトリ名。`data` ディレクトリ内にここでつけた名前のディレクトリが作成され、そこに連番で画像が保存される。
- `-i`. `--id`: id。アノテーションデータのラベルidを指定。

例) `python3 face_annotator.py -n taro -i 0`  

アプリ起動後、カメラ画像ウィンドウ上でキーボードの`s`キーを押すと、画像保存が有効化される。  
保存有効化後、顔が認識されると、その際の画像と認識した顔の位置のアノテーションファイルが指定したラベルidで保存される。  
**顔が複数認識された場合は、一番近い顔のみを保存する。認識が一時的に切れた場合など、他の人の顔に認識が移ってしまう可能性もあるため、基本的に複数の顔が視野内に入らない環境での使用を推奨**  
再度`s`キーを押すと、保存が停止する。  
アプリを終了するときは`q`キーを押す。  

