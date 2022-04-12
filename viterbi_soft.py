import numpy as np
import csv
from scipy import special
from scipy.special import comb
import math
import matplotlib.pyplot as plt
from turtle import color

R = 1/2
S_REG = 3 # レジスタ数
LENGTH = 259 # 符号長
OUTPUT_BIT = 2
LT3 = LENGTH * OUTPUT_BIT
TEST = 1000 # テスト回数

# 初期化
tdata = rdata = rdata_noncode = np.zeros((TEST, LENGTH), dtype= int)
tcode = rcode = outcode = np.zeros((TEST, LT3), dtype= int)
transmit = receive = np.zeros((TEST, LT3))
transmit_noncode = np.zeros((TEST,LENGTH))
array = [['SNR', 'BER']]

# CSV init
path = '/content/drive/MyDrive/Viterbi/'  
name = 'limits.csv'

# Ouputs
snr_list = []
ber_list = []
ber_noncode_list = []
limits = []

# tdata: 符号化前の送信データ
# tcode: 符号化後の送信データ
# rdata: 復号化前の受信データ
# rcode: 復号化後の受信データ
# transmit: 送信信号
# receive: 受信信号

def awgn(SNRdB, size):
    # awgnを作る
    No = OUTPUT_BIT * 10**(-SNRdB/10)
    return np.random.normal( 0, np.sqrt(No/2),size)+ 1j* np.random.normal( 0, np.sqrt(No/2),size)


def encoder(tdata):
    # Register state : r (bits)
    # Input : x,OUTPUT : c
    r = 0
    rs = S_REG
    for row in range(tdata.shape[0]):
        for col in range(tdata.shape[1]):
            xn = tdata[row][col]
            x = [xn]
            for i in range(rs):
                b = (r>>(rs-i-1)) & 0x01
                x.append(b)
            c1 = (x[0] + x[1] + x[3]) & 0x01
            c0 = (x[0] + x[1] + x[2] + x[3])& 0x01
            r = (r >> 1) | (xn << (rs-1))
            outcode[row][2*col] = c1
            outcode[row][2*col+1] = c0
    return outcode.astype(int).reshape(TEST,LT3)


class Status:
  def __init__(self,x):
    self.index = x
    self.pre_dist = -100
    self.next_dist = -100
    self.flag = False
    self.passed = False
    self.pre_path = []
    self.next_path = []

def forward(rcode,addr,outcode,status,current,signals,h):
    
    receive = np.array([rcode])
    channel_parameters = np.array([h])
    metric = outcode * receive * channel_parameters
    metric = metric.sum(1)

    for j in range(len(addr)):
        source = status[current[j]]
        if(source.flag == True):
            dest_status = status[addr[j]]
            new_dist = source.pre_dist + metric[j]
            dest_status.flag = True

            if(dest_status.passed == False):
                #create a new path
                now_path = source.pre_path
                tmp = now_path.copy()
                tmp.append(signals[j%2])
                new_path = tmp
                #replace
                dest_status.next_dist = new_dist
                dest_status.next_path = new_path
                dest_status.passed = True
            else:
                dest_status.passed = False
                if(dest_status.next_dist < new_dist):
                    #create a new path
                    now_path = source.pre_path
                    tmp = now_path.copy()
                    tmp.append(signals[j%2])
                    new_path = tmp
                    #replace
                    dest_status.next_dist = new_dist
                    dest_status.next_path = new_path


def status_init():
    s000 = Status(0)
    s001 = Status(1)
    s010 = Status(2)
    s011 = Status(3)
    s100 = Status(4)
    s101 = Status(5)
    s110 = Status(6)
    s111 = Status(7)
    status = [s000,s001,s010,s011,s100,s101,s110,s111]
    s000.flag = True
    s000.pre_dist = 0

    return status

def upper_limit(SNRdB):
    ck = [2,7,18,49,130,333,836,2069]
    p = math.erfc(math.sqrt( R * 10**(SNRdB/10) ))/2
    sum = 0
    for k in range(6,14):
        Pk = 0
        if(k%2==0):
            for e in range(k//2+1,k+1):
                Pk += comb(k,e) * p**e * (1-p)**(k-e)
            Pk +=  (comb(k,k//2) * p**(k//2) * (1-p)**(k//2))/2
        else:
            for e in range((k+1)//2,k+1):
                Pk += comb(k,e) * p**e * (1-p)**(k-e)
        
        sum += Pk*(ck[k-6])
    
    return sum
    

    
if __name__ == '__main__':

    #Decode Parameters
    outputs = np.array([[-1,-1],[1,1],[1,1],[-1,-1],[-1,1],[1,-1],[1,-1],[-1,1],[1,1],[-1,-1],[-1,-1],[1,1],[1,-1],[-1,1],[-1,1],[1,-1]])
    current = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7])
    signals = [0,1]
    nexts = np.array([0,1,1,2,2,3,3,4,-4,-3,-3,-2,-2,-1,-1,0])
    addr = current + nexts

    # 伝送シミュレーション
    for SNRdB in np.arange(0, 6.25, 0.25):
        # 送信データの生成
        tdata = np.random.randint(0, 2, (TEST, LENGTH - S_REG))
        tdata = np.append(tdata,np.zeros((TEST,S_REG),dtype= int),axis=1)

        # 畳み込み符号化
        tcode = encoder(tdata)

        # BPSK変調
        transmit[tcode == 0] = -1
        transmit[tcode == 1] = 1

        # チャネル係数
        h_channel = np.random.rayleigh(scale = 1,size = (TEST,LT3))
        h_nocode = np.random.rayleigh(scale = 1,size = (TEST,LENGTH))

        # 伝送
        receive_awgn_ray = h_channel * transmit + awgn(SNRdB, (TEST, LT3))
        receive_awgn = transmit + awgn(SNRdB, (TEST, LT3))
        #receive_noncode_awgn_ray = h_noncode * transmit_noncode + awgn(SNRdB,(TEST, LENGTH))
        #receive_noncode_awgn = transmit_noncode + awgn(SNRdB,(TEST, LENGTH))
        

        # ビタビ復号
        h_channel = h_channel.reshape(TEST,LENGTH,2)
        receive_awgn_ray = receive_awgn_ray.reshape(TEST,LENGTH,2)

        for j in range(TEST):
            status = status_init()
            for i,rcode in enumerate(receive_awgn_ray[j]):
                forward(rcode,addr,outputs,status,current,signals,h_channel[j][i])
                for k in status:
                  k.pre_dist = k.next_dist
                  k.next_dist = -100
                  k.pre_path = k.next_path
            #TESTごとに追加
            tmp = np.array(status[0].pre_path)
            rdata[j] = tmp
        rdata.reshape(TEST,LENGTH)
  
        # 誤り回数計算
        ok = np.count_nonzero(rdata == tdata)
        error = rdata.size - ok

        # BER計算
        BER = error / (ok + error)

        #理論上界
        #UPPER = upper_limit(SNRdB)

        snr_list.append(SNRdB)
        ber_list.append(BER)

        # 結果表示
        print(
            "SNR: {0:.2f}, BER: {1:.4e}".format(
                SNRdB, BER
            )
        )

        array.append([SNRdB, BER])
        plt.plot(snr_list, ber_list, label="simulation(hard)", color="blue")
        plt.yscale("log")