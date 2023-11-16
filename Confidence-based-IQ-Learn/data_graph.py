import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt

##### 散点密度图
from matplotlib.colors import LogNorm
from numpy import mean
from mpl_toolkits.mplot3d import axes3d
from csv import reader
from pandas import read_csv

########################### IQ_CAIL data_graph
def main(args):
    print("------------------------------------CAIL expert_demo--------------------------------------------")
    demo_file = args.demo_file
    demo_nums = args.demo_nums
    conf_file = args.conf_file
    interval = args.interval
    pic_nums = args.pic_nums
    subopt_class_num = args.subopt_class_num


    # 导入专家演示数据并打印基本信息
    with open(demo_file, "rb") as c:
        iq_cail = pickle.load(c)
        iq_cail = dict(iq_cail)
        print("info.keys():", iq_cail.keys())

    # 记录专家演示样本的奖励信息
    mean_return = []
    for i in range(demo_nums):
        epoch_return = np.array(iq_cail["rewards"][i])
        # print("each_return:", epoch_return.sum())
        mean_return.append(epoch_return.sum())


    for i in range(pic_nums): # 导入经过多少的step的模型conf
        step_num = i*interval+1
        conf_file_path = conf_file + f"/config/{step_num}_conf.csv"

        # 导入conf数据
        with open(conf_file_path,"rt") as raw_data:
            readers = reader(raw_data, delimiter="\n")
            x = list(readers)
            data = np.array(x).astype('float')
            # print(data.shape)
            # print(type(data))


        ####################### 绘制三维散点图


        ####################### 数据
        x = []
        y = []
        z = []
        colr = []
        p = 1
        for i in range(0, len(data), p):
            x.append(i)
            z.append(float(data[i]))
            colr.append(float(data[i]))
            for j in range(len(data)//1000):
                # 把样本中每一条轨迹分开切片
                if i<(j+1)*1000 and i>=j*1000:
                    y.append(mean_return[j])

                # if i<(j+1)*8000*5 and i>=j*8000*5:
                #     colr.append(j)
                


        ####################### 绘图
        ###### 三维散点图
        # fig, ax = plt.subplots(figsize=(7,5)) #figuresize图片比例
        plt.figure("3D Scatter", facecolor="lightgray") # 在一个窗口显示
        ax3d = plt.gca(projection="3d")  # 创建三维坐标

        plt.title('3D Scatter')
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('z')
        # plt.tick_params(labelsize=10)

        scatter = ax3d.scatter(x, y, z, s=0.01, c=colr, cmap='jet', marker="o")
        if not os.path.exists(conf_file + '/conf_pic'):
            os.mkdir(conf_file + '/conf_pic')
        plt.savefig(conf_file + '/conf_pic/' + f'3D_step_{step_num}.png')
        # plt.close()


        length = subopt_class_num*1000
        clips_return = []
        # print("len(data):", len(data))
        for i in range(0, len(data)//length, 1):
            clips_return.append(mean(mean_return[i*subopt_class_num*8:(i+1)*subopt_class_num*8]))
            plt.figure(f"{i}_{clips_return[i]}") # 在一个窗口显示
            # normal distribution center at x=0 and y=5
            ax = x[i*length//p:(i+1)*length//p]
            ay = z[i*length//p:(i+1)*length//p]

            plt.hist2d(ax, ay, bins=500, norm=LogNorm())
            plt.colorbar()
            plt.savefig(conf_file + '/conf_pic/' + f'step_{step_num}' + f'_{clips_return[i]}_density.png')
            plt.close()

    # plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--demo_file', type=str, required=True,
                   help='path to the demonstration')
    p.add_argument('--conf_file', type=str, required=True,
                   help='path to the confidence')

    # custom
    p.add_argument('--demo_nums', type=int, default=200,
                   help='demonstration nums')
    p.add_argument('--interval', type=int, default=4000,
                   help='interval nums')
    p.add_argument('--pic_nums', type=int, default=5,
                   help='pic nums')
    p.add_argument('--subopt_class_num', type=int, default=5,
                   help='subopt_class_num')

    args = p.parse_args()
    main(args)