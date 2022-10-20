import os

if __name__ == '__main__':
    cmd =  'export PYTHONPATH="/home/lbyang/workspace/HDN_mindspore:/home/lbyang/workspace/HDN_mindspore/homo_estimator/Deep_homography/Oneline_DLTv2:"'
    os.system(cmd)
    cmd = 'export PATH="/home/lbyang/anaconda3/envs/hdn_mindspore/bin:/home/lbyang/install/obsutil_linux_amd64_5.4.6:/usr/local/cuda-11.2/bin:/home/lbyang/anaconda3/condabin:/home/lbyang/anaconda3/condabin:/home/lbyang/bin:/usr/local/bin:/home/lbyang/bin:/usr/local/bin:/home/lbyang/bin:/usr/local/bin:/home/lbyang/bin:/usr/local/bin:/home/lbyang/.vscode-server/bin/57fd6d0195bb9b9d1b49f6da5db789060795de47/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/lbyang/anaconda3/envs/hdn_mindspore/bin"'
    os.system(cmd)
    videoList = ['V02_6', 'V02_7', 'V03_4', 'V03_6', 'V04_5', 'V05_3', 'V06_1', 'V07_4', 'V08_1', 'V08_4', 'V10_5', 'V12_4', 'V14_7', 'V15_7', 'V18_4', 'V21_3']
    #all video
    # for i in range(2, 31):
    #     for j in range(1, 8):
    #         videoList.append('V{:02.0f}_{}'.format(i, j))
    # print(videoList)

    for video in videoList:
        cmd = 'python ../../tools/test.py --snapshot ../../hdn.ckpt --config proj_e2e_GOT_unconstrained_v2.yaml --dataset POT210 --video '+ video
        resultState = os.system(cmd)
        while resultState != 0:
            resultState = os.system(cmd)
        evalCmd = 'python ../../tools/eval.py --tracker_path ./results --dataset POT210 --num 1 --tracker_prefix hdn --video '+ video
        os.system(evalCmd)
