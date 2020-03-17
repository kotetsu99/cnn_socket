#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.preprocessing import image
import numpy as np
import socket
import os, sys


# サーバーIPアドレス定義
host = "0.0.0.0"
# サーバーの待ち受けポート番号定義
port = 50001

# 入力画像リサイズ定義
img_width, img_height = 64, 107

# 学習用画像ファイル保存ディレクトリ
train_data_dir = 'dataset/train'
# 受信画像保存ディレクトリ
sc_dir = 'dataset/sc'
# 受信画像ファイル名
sc_file = 'sc_file.png'

def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    # データセットのサブディレクトリ名（クラス名）を取得
    classes = []
    for d in os.listdir(train_data_dir):
        if os.path.isdir(os.path.join(train_data_dir, d)):
            classes.append(d)
    print 'クラス名リスト = ', classes

    # 学習済ファイルの確認
    if not len(sys.argv)==2:
        print('使用法: python cnn_server.py 学習済ファイル名.h5')
        sys.exit()
    savefile = sys.argv[1]

    # モデルのロード
    model = keras.models.load_model(savefile)

    # ソケット定義(IPv4,TCPによるソケット)
    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # TIME-WAIT状態が切れる前にソケットを再利用
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # IPとPORTを指定してバインド
    serversock.bind((host,port))
    # ソケット接続待受（キューの最大数を指定）
    serversock.listen(10)

    while True:
        # ソケット接続受信待ち
        try:
            print 'クライアントからの接続待ち...'
            # 接続されればデータを格納
            clientsock, client_address = serversock.accept()

        # 接続待ちの間に強制終了が入った時の例外処理
        except KeyboardInterrupt:
            print 'Ctrl + C により強制終了'
            break

        # 接続待ちの間に強制終了なく、クライアントからの接続が来た場合
        else:
            # 画像保存先ファイルオープン
            f =  open(sc_dir + '/' + sc_file, 'wb')

            # ソケット接続開始後の処理
            while True:
                # データ受信。受信バッファサイズ1024バイト
                data = clientsock.recv(1024)
                # 全データ受信完了（受信路切断）時に、応答・切断処理開始
                if not data:
                    # ファイルクローズ
                    f.close()
                    # AI画像認識
                    res = cnn_recognition(model, classes)
                    # 認識結果をクライアントに送信
                    clientsock.sendall(res)
                    # コネクション切断
                    clientsock.close()
                    break
                # ファイルにデータ書込
                f.write(data)
        
    # ソケットクローズ(強制終了時に実行）
    serversock.close()


def cnn_recognition(model, classes):
    # 画像ファイル取得
    filename = os.path.join(sc_dir, sc_file)
    img = image.load_img(filename, target_size=(img_width, img_height))

    # 入力画像定義(画像の行, 画像の列, チャネル数の3次元テンソル)
    x = image.img_to_array(img)
    # 4次元テンソル(サンプル数, チャネル数, 画像の行数, 画像の列数)に変換
    x = np.expand_dims(x, axis=0)
    # 学習時に正規化してるので、ここでも正規化
    x = x / 255
    # 画像サンプルが1枚のみなので、最初の1枚[0]の認識結果を格納
    pred = model.predict(x)[0]

    # 予測確率が高いトップを出力
    # 今回は最も似ているクラスのみ出力したいので1にしているが、上位n個を表示させることも可能。
    top = 3
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    print('file name is', sc_file)
    print(result)
    print('=======================================')
    return result[0][0]


if __name__ == '__main__':
    main()
