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
LT3 = int(LENGTH * 1/R)
TEST = 100 # テスト回数
OUTPUT_BIT = 2

# 初期化
tdata = rdata = rdata_noncode = np.zeros((TEST, LENGTH), dtype= int)
tcode = rcode = outcode = np.zeros((TEST, LT3), dtype= int)
transmit = receive = np.zeros((TEST, LT3))
transmit_noncode = np.zeros((TEST,LENGTH))
array = [['SNR', 'BER','BER_NON']]

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
    No = R * 10**(-SNRdB/10)
    return np.random.normal( 0, np.sqrt(No/2),size) + 1j* np.random.normal( 0, np.sqrt(No/2),size)

def get_output(r,xn):
    x = [xn]
    rs = S_REG
    for i in range(rs):
        b = (r>>(rs-i-1)) & 0x01
        x.append(b)
    c1 = (x[0] + x[1] + x[3]) & 0x01
    c0 = (x[0] + x[1] + x[2] + x[3])& 0x01
    new_r = (r >> 1) | (xn << (rs-1))
    return c1,c0,new_r

def encoder(tdata):
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


class Stats:
  def __init__(self,x):
    self.index = x
    self.pre_dist = 100
    self.next_dist = 100
    self.flag = False
    self.passed = False
    self.pre_path = []
    self.next_path = []


def forward(rcode,addr,outputs,stats,current,signals):
    inputs = np.array([rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode,rcode])
    distance = abs(inputs-outputs)
    distance = distance.sum(1)

    for j in range(len(addr)):
      source = stats[current[j]]
      if(source.flag == True):
          i = addr[j]
          dest = stats[i]
          new_dist = source.pre_dist + distance[j]

          now_path = source.pre_path
          tmp = now_path.copy()
          tmp.append(signals[j%2])
          new_path = tmp

          dest.flag = True
          if(dest.passed == False):
            dest.next_dist = new_dist
            dest.passed = True
            dest.next_path = new_path
          else:
            dest.passed = False
            if(dest.next_dist > new_dist):
                dest.next_dist = new_dist
                dest.next_path = new_path


def stats_init():
    s000 = Stats(0)
    s001 = Stats(1)
    s010 = Stats(2)
    s011 = Stats(3)
    s100 = Stats(4)
    s101 = Stats(5)
    s110 = Stats(6)
    s111 = Stats(7)
    stats = [s000,s001,s010,s011,s100,s101,s110,s111]
    s000.flag = True
    s000.pre_dist = 0

    return stats

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
    outputs = np.array([[0,0],[1,1],[1,1],[0,0],[0,1],[1,0],[1,0],[0,1],[1,1],[0,0],[0,0],[1,1],[1,0],[0,1],[0,1],[1,0]])
    current = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7])
    signals = [0,1]
    nexts = np.array([0,1,1,2,2,3,3,4,-4,-3,-3,-2,-2,-1,-1,0])
    addr = current + nexts

    # 伝送シミュレーション
    for SNRdB in np.arange(0, 6.25, 0.25):
        # 送信データの生成
        tdata = np.random.randint(0, 2, (TEST, LENGTH - S_REG))
        #print(tdata)
        # 終端ビット系列の付加 3bit 0
        tdata = np.append(tdata,np.zeros((TEST,3),dtype= int),axis=1)
        tdata_noncode = tdata.copy()
        # 畳み込み符号化
        tcode = encoder(tdata)
        # BPSK変調
        transmit[tcode == 0] = -1
        transmit[tcode == 1] = 1
        # 伝送
        receive = transmit + awgn(SNRdB, (TEST, LT3))
        # BPSK復調
        rcode[receive < 0] = 0
        rcode[receive >= 0] = 1
        # ビタビ復号
        for j in range(TEST):
            stats = stats_init()
            for i in rcode.reshape(TEST,LENGTH,2)[j]:
                forward(i,addr,outputs,stats,current,signals)
                for i in stats:
                  i.pre_dist = i.next_dist
                  i.next_dist = 100
                  i.pre_path = i.next_path
            #TESTごとに追加
            result  = stats[0].pre_path
            tmp = np.array(result)
            rdata[j] = tmp
        rdata.reshape(TEST,LENGTH)
  
        # 誤り回数計算
        ok = np.count_nonzero(rdata == tdata)
        error = rdata.size - ok

        # BER計算
        BER = error / (ok + error)

        # 非符号化
        transmit_noncode[tdata_noncode == 0] = -1
        transmit_noncode[tdata_noncode == 1] = 1
        receive_noncode = transmit_noncode + awgn(SNRdB,(TEST, LENGTH))
        rdata_noncode[receive_noncode < 0] = 0
        rdata_noncode[receive_noncode >= 0] = 1
        
        ok_noncode = np.count_nonzero(rdata_noncode == tdata_noncode)
        error_noncode = rdata_noncode.size - ok_noncode

        NONCODE = error_noncode / (ok_noncode + error_noncode)

        #理論上界
        UPPER = upper_limit(SNRdB)

        snr_list.append(SNRdB)
        ber_list.append(BER)
        ber_noncode_list.append(NONCODE)
        limits.append(UPPER)

        # 結果表示
        print(
            "SNR: {0:.2f}, BER: {1:.4e}, BER_NONCODE:{2:.4e}".format(
                SNRdB, BER, NONCODE
            )
        )

        array.append([SNRdB, BER,NONCODE,UPPER])
        plt.plot(snr_list, limits, label="limits", color="green")
        plt.plot(snr_list, ber_list, label="simulation(hard)", color="blue")
        plt.plot(snr_list, ber_noncode_list, label="without code", color="red")
        plt.yscale("log")
        '''
        # plt.plot(snr_list, p_b_list, label="p_b", color="green")
                
        with open(path+name,'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(array)
        '''
print(limits)
