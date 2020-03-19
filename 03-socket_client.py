#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import os, sys


# サーバーIPアドレス定義
host = "0.0.0.0"
# サーバー待ち受けポート番号定義
port = 50001


def main():

    # 送信画像ファイルパス引数の取得
    if not len(sys.argv)==2:
        print('使用法: python 03-socket_client.py 画像ファイル名')
        sys.exit()
    image_file = sys.argv[1]

    # ファイルをバイナリデータとして読み込み
    with open(image_file, 'rb') as f:
        binary = f.read()

    # ソケットクライアント作成
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # 送信先サーバーに接続
        s.connect((host, port))

        # サーバーにバイナリデータを送る
        print(image_file + 'をサーバーに送信')
        s.sendall(binary)
        # データ送信完了後、送信路を閉じる
        s.shutdown(1)

        # サーバーからの応答を取得。バッファサイズ1024バイト
        res = s.recv(1024)
        # サーバー応答を表示
        print('認識結果： ' + res)

    except Exception as e:
        # 例外が発生した場合、内容を表示
        print(e)

    finally:
        # ソケットを閉じて終了
        s.close()


if __name__ == '__main__':
    main()
