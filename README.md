## 在PyTorch中使用Transformer模型进行中文文本分类

### 数据集

sougou小分类数据集，共有5个类别，分别为体育、健康、军事、教育、汽车。划分为训练集和测试集，其中训练集每个类别800条样本，测试集每个类别100条样本。

### 模型

Transformer模型中的Encoder部分，具体实现可参考文章: [https://pytorch.org/tutorials/beginner/transformer_tutorial.html](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)



### 模型效果

在测试集上的评估指标:

```
              precision    recall  f1-score   support

          体育     0.8679    0.9293    0.8976        99
          健康     0.8257    0.9091    0.8654        99
          军事     1.0000    0.9293    0.9634        99
          教育     0.9451    0.8687    0.9053        99
          汽车     0.8866    0.8687    0.8776        99

    accuracy                         0.9010       495
   macro avg     0.9051    0.9010    0.9018       495
weighted avg     0.9051    0.9010    0.9018       495
```

混淆矩阵:

![混淆矩阵](https://raw.githubusercontent.com/percent4/pytorch_transformer_chinese_text_classification/master/image/confusion_matrix.png)

在新样本上的表现:

> 盖世汽车讯，特斯拉去年击败了宝马，夺得了美国豪华汽车市场的桂冠，并在今年实现了开门红。1月份，得益于大幅降价和7500美元美国电动汽车税收抵免，特斯拉再度击败宝马，蝉联了美国豪华车销冠，并且注册量超过了排名第三的梅赛德斯-奔驰和排名第四的雷克萨斯的总和。根据Experian的数据，在所有豪华品牌中，1月份，特斯拉在美国的豪华车注册量为49，917辆，同比增长34%；宝马的注册量为31，070辆，同比增长2.5%；奔驰的注册量为23，345辆，同比增长7.3%；雷克萨斯的注册量为23，082辆，同比下降6.6%。奥迪以19，113辆的注册量排名第五，同比增长38%。凯迪拉克注册量为13，220辆，较去年同期增长36%，排名第六。排名第七的讴歌的注册量为10，833辆，同比增长32%。沃尔沃汽车排名第八，注册量为8，864辆，同比增长1.8%。路虎以7，003辆的注册量排名第九，林肯以6，964辆的注册量排名第十。|汽车|

> 预测标签：汽车

> 北京时间3月16日，NBA官方公布了对于灰熊球星贾-莫兰特直播中持枪事件的调查结果灰熊，由于无法确定枪支是否为莫兰特所有，也无法证明他曾持枪到过NBA场馆，因为对他处以禁赛八场的处罚，且此前已禁赛场次将算在禁赛八场的场次内，他最早将在下周复出。

> 预测标签：体育

> 3月11日，由新浪教育、微博教育、择校行联合主办的“新浪&微博2023国际教育春季巡展•深圳站”于深圳凯宾斯基酒店成功举办。深圳优质学校亮相展会，上千组家庭前来逛展。近30所全国及深圳民办国际化学校、外籍人员子女学校、公办学校国际部等多元化、多类型优质学校参与了本次活动。此外，近10位国际化学校校长分享了学校的办学特色、教育理念及学生的成长案例，参展家庭纷纷表示受益匪浅。展会搭建家校沟通桥梁，帮助家长们合理规划孩子的国际教育之路。深圳国际预科书院/招生办主任沈兰Nancy Shen参加了本次活动并带来了精彩的演讲，以下为演讲实录：

> 预测标签：教育

> 据中国、伊朗、俄罗斯等国军队共识，3月15日至19日，中、伊、俄等国海军将在阿曼湾举行“安全纽带-2023”海上联合军事演习。“安全纽带-2023”海上联演由2019年、2022年先后两次举行的中伊俄海上联演发展而来，中方派出南宁号导弹驱逐舰参演，主要参加空中搜索、海上救援、海上分列式等科目演练。此次演习有助于深化参演国海军务实合作，进一步展示共同维护海上安全、积极构建海洋命运共同体的意愿与能力，为地区和平稳定注入正能量。

> 预测标签：军事

> 指导专家：皮肤科教研室副主任、武汉协和医院皮肤性病科主任医师冯爱平教授在临床上，经常能看到有些人出现反复发作的口腔溃疡，四季不断，深受其扰。其实这已不单单是口腔问题，而是全身疾病的体现，特别是一些免疫系统疾病，不仅表现在皮肤还会损害黏膜，下列几种情况是造成“复发性口腔溃疡”的原因。缺乏维生素及微量元素。缺乏微量元素锌、铁、叶酸、维生素B12等时，会引发口角炎。很多日常生活行为可能造成维生素的缺乏，如过分淘洗米、长期进食精米面、吃素食等，很容易造成B族维生素的缺失。

> 预测标签：健康

### 参数影响

我们考察模型参数对模型在验证集上的表现的影响。

- 考察句子长度对模型表现的影响

保持其它参数不变，设置文本长度（SENT_LENGTH）分别为200，256，300，结果如下：

| 文本长度 |  accuracy|precision|recall|f1-score|
|--|--|---|---|---|
|200|0.9010|0.9051|0.9010|0.9018|
|256|0.8990|0.9019|0.8990|0.8977|
|300|0.8788|0.8824|0.8788|0.8774|

- 考察词向量维度对模型表现的影响

设置文本长度（SENT_LENGTH）为200，保持其它参数不变，设置词向量维度为32, 64， 128，结果如下：

| 词向量维度|  accuracy|precision|recall|f1-score|
|--|--|---|---|---|
|32|0.6869|0.7402|0.6869|0.6738|
|64|0.7576|0.7629|0.7576|0.7518|
|128|0.9010|0.9051|0.9010|0.9018|
|256|0.9212|0.9238|0.9212|0.9213|
