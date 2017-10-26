
# Machine Learning Project Checklist

1. Frame the problem and look at the big picture. 将问题框架化并且关注重点
2. Get the data. 获取数据
3. Explore the data to gain insights. 探索数据从而洞悉数据 
4. Prepare the data to better expose the underlying data patterns to Machine Learning algorithms. 准备数据从而更好地将底层数据模式提供给机器学习算法
5. Explore many different models and short-list the best ones. 探索许多不同的模型并列出最好的那些
6. Fine-tune your models and combine them into a great solution. 调整模型并将它们结合为很棒的解决方案
7. Present your solution. 展示你的解决方案 
8. Launch, monitor, and maintain your system 建立，监督并维护你的系统

---

## 1. Frame the Problem and Look at the Big Picture

1. Define the objective in business terms. 用商务术语定义目标
2. How will your solution be used? 你的解决方案怎样被使用？
3. What are the current solutions/workarounds (if any)? 当前的解决方案/解决方法是什么（如果有的话）？
4. How should you frame this problem (supervised/unsupervised, online/offline,etc.)? 如何框架化问题？（监督/无监督？在线/离线？）
5. How should performance be measured? 如何度量模型的表现？
6. Is the performance measure aligned with the business objective? 方案的表现是否和商业目标一致？ 
7. What would be the minimum performance needed to reach the business objective? 
8. What are comparable problems? Can you reuse experience or tools? 类似的问题如何解决？是否可以重复使用经验或工具？
9. Is human expertise available? 人员是否专业？
10. How would you solve the problem manually? 你如何动手解决问题？ 
11. List the assumptions you (or others) have made so far. 列出你目前所做的假设
12. Verify assumptions if possible. 如果可能，请验证假设

## 2. Get the Data

Note: automate as much as possible so you can easily get fresh data. 

1. List the data you need and how much you need.  列出你需要的数据和数据量
2. Find and document where you can get that data. 寻找并记录哪里可以得到那些数据
3. Check how much space it will take. 核对它需要多大的存储空间
4. Check legal obligations, and get authorization if necessary. 核对法律义务并在必要时获取授权
5. Get access authorizations. 获取使用授权
6. Create a workspace (with enough storage space). 创建工作目录（拥有足够的存储空间）
7. Get the data. 获取数据
8. Convert the data to a format you can easily manipulate (without changing the data itself). 将数据转换为你可以容易操作的格式（不改变数据源本身的情况下）
9. Ensure sensitive information is deleted or protected (e.g., anonymized). 确保敏感信息被删除或保护（比如匿名化）
10. Check the size and type of data (time series, sample, geographical, etc.). 核对数据的大小和类型（时间序列、样本、地理信息等）
11. Sample a test set, put it aside, and never look at it (no data snooping!). 抽样出测试集，将它放在一边，以后不需要关注它（不是数据窥探！）

## 3. Explore the Data

Note: try to get insights from a field expert for these steps. 注意：尝试从领域专家那获取这些步骤的见解

  1. Create a copy of the data for exploration (sampling it down to a manageable size if necessary). 创建数据副本用来探索（如有必要，将其取样至可管理的大小）
  2. Create a Jupyter notebook to keep a record of your data exploration. 创建一个 Jupyter notebook 用来保持记录你的数据探索的过程
  3. Study each attribute and its characteristics: 研究每个属性和它的特点
     * Name 名称
     * Type (categorical, int/float, bounded/unbounded, text, structured, etc.) 类型（种类，整形/浮点型，有界/无界，文本，结构化数据等）
     * % of missing values 缺失数据的百分比
     * Noisiness and type of noise (stochastic, outliers, rounding errors, etc.) 噪声点和它的类型（随机点，异常点，舍入误差等）
     * Possibly useful for the task? 对任务可能有用吗？
     * Type of distribution (Gaussian, uniform, logarithmic, etc.) 分布类型（高斯分布，联合分布，对数分布等）
  4. For supervised learning tasks, identify the target attribute(s). 对于监督学习任务，标识目标属性值
  5. Visualize the data. 可视化数据
  6. Study the correlations between attributes. 研究属性间的相关性
  7. Study how you would solve the problem manually. 研究怎样手工解决问题
  8. Identify the promising transformations you may want to apply. 确定你想要应用的有效的转换
  9. Identify extra data that would be useful 确定出也许要用的额外数据
  10. Document what you have learned. 用文档记录你所了解到的东西

## 4. Prepare the Data

Notes: 

* Work on copies of the data (keep the original dataset intact). 在数据副本上进行工作（保持原来的数据集完好无损）
* Write functions for all data transformations you apply, for five reasons: 写出所有你要应用数据转换的函数，因为以下五点原因：

  * So you can easily prepare the data the next time you get a fresh dataset 当你下次拿到一个新的数据集时可以容易地准备数据
  * So you can apply these transformations in future projects 你可以在未来的项目中应用这些转换
  * To clean and prepare the test set 用来清洗并准备测试数据集 
  * To clean and prepare new data instances once your solution is live 一旦项目上线可以用来清洗和准备新的数据集
  * To make it easy to treat your preparation choices as hyperparameters 

1. Data cleaning: 数据清洗：

   * Fix or remove outliers (optional). 修正或移除异常点（可选）
   * Fill in missing values (e.g., with zero, mean, median…) or drop their rows (orcolumns). 填补缺失值（比如用零，平均值，中位数...）

2. Feature selection (optional): 特征提取（可选）：

   * Drop the attributes that provide no useful information for the task. 丢弃无法为任务提供有用信息的特征

3. Feature engineering, where appropriate: 特征工程

   * Discretize continuous features. 连续特征离散化
   * Decompose features (e.g., categorical, date/time, etc.). 分解特征（比如划分种类，日期/时间等）
   * Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.). 对特征添加有益的转换（比如 log(x), sqrt(x), x^2等）
   * Aggregate features into promising new features. 将一些特征融合为有益的新特征

4. Feature scaling: standardize or normalize features. 特征规约：将特征标准化、规范化

## 5. Short-List Promising Models

Notes: 

* If the data is huge, you may want to sample smaller training sets so you can train many different models in a reasonable time (be aware that this penalizes complexmodels such as large neural nets or Random Forests). 如果数据非常大，你可以抽样出小的训练集从而训练许多不同的模型
* Once again, try to automate these steps as much as possible. 一次又一次，尝试将这些步骤尽可能地自动化处理

1. Train many quick and dirty models from different categories (e.g., linear, naiveBayes, SVM, Random Forests, neural net, etc.) using standard parameters. 使用标准参数训练快速、粗糙的模型（比如线性模型，朴素贝叶斯，支持向量机，随机森林，神经网络等）
2. Measure and compare their performance. 衡量并比较他们的表现
   
   * For each model, use N-fold cross-validation and compute the mean and standarddeviation of the performance measure on the N folds. 对于每个模型使用N折交叉验证法，并且计算基于N折的均值与方差

3. Analyze the most significant variables for each algorithm. 分析对于每个算法最重要的变量
4. Analyze the types of errors the models make.分析模型产生的错误类型

   * What data would a human have used to avoid these errors? 

5. Have a quick round of feature selection and engineering. 进行一轮快速的特征提取和特征工程
6. Have one or two more quick iterations of the five previous steps. 对之前的五个步骤进行一至两次的快速遍历
7. Short-list the top three to five most promising models, preferring models that make different types of errors. 列出前三到五名最有用的模型

## 6. Fine-Tune the System

Notes: 

* You will want to use as much data as possible for this step, especially as you movetoward the end of fine-tuning. 这一步你将会使用尽可能多的数据，特别是当你进行到这一步的最后阶段时
* As always automate what you can. 像之前一样尽可能让你做的自动化

1. Fine-tune the hyperparameters using cross-validation. 使用交叉验证方法调节超参数
  
   * Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., should I replace missing values with zero or with the median value? Or just drop the rows?). 要像调节超参数那样对待数据转换的过程，特别是当你不知如何下手的时候（比如，我应该是用平均值还是中位数替换缺失值？或者还是直接丢弃它们？）
   * Unless there are very few hyperparameter values to explore, prefer randomsearch over grid search. If training is very long, you may prefer a Bayesianoptimization approach (e.g., using Gaussian process priors). 除非只有很少的超参数，最好使用随机搜索而非网格搜索。如果训练的时间很长，你应该使用贝叶斯优化方法（比如，使用高斯处理）

2. Try Ensemble methods. Combining your best models will often perform better than running them individually. 尝试集成方法，将模型结合起来往往比单独运行他们表现地要好
3. Once you are confident about your final model, measure its performance on the test set to estimate the generalization error. 一旦你对你最终的模型很有自信，在测试集上测试以衡量它的表现来估计泛化误差

## 7. Present Your Solution

1. Document what you have done. 将你做的工作整理成文档
2. Create a nice presentation. 制作精美的展示
   
   * Make sure you highlight the big picture first. 确保你首先突出重点

3. Explain why your solution achieves the business objective. 解释你的方案为什么可以获得商业目标
4. Don’t forget to present interesting points you noticed along the way. 不要忘记展示在这过程中你注意到的有趣的点

   * Describe what worked and what did not. 描述哪些奏效，哪些不奏效
   * List your assumptions and your system’s limitations. 列出你的假设和系统的限制

5. Ensure your key findings are communicated through beautiful visualizations or easy-to-remember statements (e.g., “the median income is the number-one predictor of housing prices”). 确保你的主要发现是通过漂亮的可视化方式或容易记住的语言来表达的（例如，“平均收入是房价预测的头号预测者”）

## 8. Launch!

1. Get your solution ready for production (plug into production data inputs, write unit tests, etc.). 使你的解决方案准备好用于生产环境（）
2. Write monitoring code to check your system’s live performance at regular intervalsand trigger alerts when it drops. 
   
   * Beware of slow degradation too: models tend to “rot” as data evolves. 时时刻刻意识到随着数据的进入模型会逐渐变得很差
   * Measuring performance may require a human pipeline (e.g., via a crowdsourcing service). 评估模型可能需要大量的人力（比如，通过众包服务可以解决这个问题） 
   * Also monitor your inputs’ quality (e.g., a malfunctioning sensor sending random values, or another team’s output becoming stale). This is particularly important for online learning systems. 同时监控输入数据的质量（例如，一个有故障的传感器发送随机数，或者另外一个团队的输出变得陈旧），这对于在线学习系统尤其重要

3. Retrain your models on a regular basis on fresh data (automate as much as possible). 定期对新数据再次训练你的模型（尽可能自动化）
