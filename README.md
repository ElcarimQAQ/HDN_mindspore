hdn 的mindspore版本
原项目：https://github.com/zhanxinrui/HDN

```bash
 cd experiments/tracker_homo_config/
```

配置文件：proj_e2e_GOT_unconstrained_v2.yaml

# test

```bash
python ../../tools/test.py --snapshot ../../hdn.ckpt --config proj_e2e_GOT_unconstrained_v2.yaml --dataset POT210 --video
```

建议使用muti_test:

```bash
python ../../tools/test.py
```

test 精度基本达标（需要跑多次）：

![image-20221015222450071](http://picbed.elcarimqaq.top/img/image-20221015222450071.png)

TODO:

- [ ] 多次test 不一致问题 ：https://bbs.huaweicloud.com/forum/thread-0293997503678600024-1-1.html

- [ ] ops.gather 疑似bug ，结果不一致问题：https://bbs.huaweicloud.com/forum/thread-0246100173357120053-1-1.html 

  只有在代码中会出现，单独有ops.gether 跑下面数据并不会出现，怀疑是否和动态shape有关？

  使用数据为：https://github.com/ElcarimQAQ/HDN_mindspore/blob/master/training_dataset/trainDictData.json

- [ ] triplet loss 问题：https://bbs.huaweicloud.com/forum/thread-0202978484140620007-1-1.html

  目前mindspore 版本是有bug的，暂时用我自己的myTripletMarginLoss（不支持batchsize不为1） 代替，等待mindspore完善

- [ ] train脚本问题梯度下降问题，在ops.gather 算子处报错

  ![image-20221015223205745](http://picbed.elcarimqaq.top/img/image-20221015223205745.png)

