

- [ ] 多次test 不一致问题 ：https://bbs.huaweicloud.com/forum/thread-0293997503678600024-1-1.html

- [ ] ops.gather 疑似bug ，结果不一致问题：https://bbs.huaweicloud.com/forum/thread-0246100173357120053-1-1.html 

  只有在代码中会出现，单独有ops.gether 跑下面数据并不会出现，怀疑是否和动态shape有关？

  使用数据为：https://github.com/ElcarimQAQ/HDN_mindspore/blob/master/training_dataset/trainDictData.json

- [ ] triplet loss 问题：https://bbs.huaweicloud.com/forum/thread-0202978484140620007-1-1.html

  目前mindspore 版本是有bug的，暂时用我自己的myTripletMarginLoss（不支持batchsize不为1） 代替，等待mindspore完善

- [ ] train脚本在梯度下降中存在问题，在ops.gather 算子处报错

  ![image-20221015223205745](http://picbed.elcarimqaq.top/img/image-20221015223205745.png)

- [ ] cls_loss_lp_sup，loc_loss_lp_sup 会引起梯度下降问题

  RuntimeError: For 'LogSoftmaxGrad', the dimension of input must be equal to 2, but got 4

  ![image-20221015234111174](http://picbed.elcarimqaq.top/img/image-20221015234111174.png)

  在同一组数据下，保留cls_loss_sup ， loc_loss_sup ，cls_loss_sup 误差很大，其他相差无几

- [ ] 自己写的TrainOneStepCell没有self.network.set_grad() 就没法顺利执行，exit code 139