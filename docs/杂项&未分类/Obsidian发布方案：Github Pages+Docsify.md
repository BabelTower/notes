# 0. 前言

最近（2021-12-11）发现Obsidian+Docsify+Github Pages这一套搭配很适合来建立（线上的）个人知识库，而且都是免费的。

这一套搭配的优点有：
- 所有的工具、资源都是免费的
- docsify生成文档网站非常简单，只需要创建一个 `index.html`
- 相比于“文件夹+Markdown”，Obsidian在保证了本地化的同时，提供了更高效的文档组织管理方式

这一套搭配的缺点有：
- 前期配置和学习使用有时间成本
- Github Pages没啥流量，搭建站点是自娱自乐

先来看看效果展示图（你可以做更多你喜欢的DIY）：

![首页截图](assets/截屏2021-12-11%20下午4.29.35.png)

![文档截图](assets/截屏2021-12-11%20下午4.33.26.png)

![Obsidian编辑页面](assets/截屏2021-12-11%20下午4.58.50.png)


# 1. 相关资源汇总

-   [docsify 中文文档](https://docsify.js.org/#/zh-cn/)
-   [我的Github示例项目](https://github.com/BabelTower/notes)
-   docsify [教程1](https://zhuanlan.zhihu.com/p/101126727)、[教程2](https://zhuanlan.zhihu.com/p/70219397)
-   obsidian [知乎精华回答](https://www.zhihu.com/topic/21349840/top-answers)

# 2. 快速实现

![Drawing 2021-12-11 17.18.44.excalidraw](../Excalidraw/Drawing%202021-12-11%2017.18.44.excalidraw.png)

基础选项/乞丐版：
1. 下载Obsidian软件，将vault建立在你的云盘（icloud）文件夹下。
2. 直接在云盘vault文件夹下初始化docsify（不要选择新建docs目录）。
3. 在本地建立一个文件夹，复制云盘文件夹下所有文件到本地文件夹的docs目录。
4. 在Github建立新的repo，将你的本地文件夹上传，设置该repo的docs目录为Github page的Source。

进阶选项：
1. 支持网页图片：
	1. 取消“设置->文件链接->使用wiki链接”这一项。
	2. “设置->文件链接”中新附件的位置为“当前文件夹的子文件夹”。
2.  一些常用命令写成bash脚本
	1.  用cp命令拷贝你的vault文件夹，参考我写的bash脚本[cp.sh](https://github.com/BabelTower/notes/blob/main/cp.sh)
	2.  git push也可写成脚本[push.sh](https://github.com/BabelTower/notes/blob/main/push.sh)

# 3. 后话

如果有人看，再补充实现细节。