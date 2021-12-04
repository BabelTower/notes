
## pandoc命令

必选：
`--pdf-engine=xelatex -V mainfont='STXihei'`
这里要选用Mac中文字体，更多见[英文版 Mac 上 office 的中文字体](https://www.jianshu.com/p/8cf09c5144e2)

-template=pm-template.latex

论文引用：
`--cite --bibliography=我的笔记/_media/export.bib`

可选：
加上目录
`--toc `

各个 section 加上编号
`-N`

更改 PDF 的 margin，使用默认设置生成的 PDF margin 太大，[根据 Pandoc 官方 FAQ](https://pandoc.org/faqs.html#how-do-i-change-the-margins-in-pdf-output)， 可以使用下面的选项更改 margin：

`-V geometry:"top=2cm, bottom=1.5cm, left=2cm, right=2cm"`

## 示例 

将[HAR模型预测金融数据波动率](../../Projects/Volatility%20Forecasting/HAR模型预测金融数据波动率.md)导出为pdf的示例：

```bash
pandoc --pdf-engine=xelatex \
--template=../../_media/template.tex \
--cite --bibliography=../../_media/export.bib -M reference-section-title="参考文献" \
--csl=../../_media/china-national-standard-gb-t-7714-2015-author-date.csl \
HAR模型预测金融数据波动率.md -o output.pdf
```

## 教程
1. [[转载]使用 Pandoc 与 Markdown 生成 PDF 文件](https://www.cnblogs.com/charleechan/p/12319264.html)
2. [利用Pandoc将markdown文件转化为pdf](https://www.cnblogs.com/loongfee/p/3223957.html)
3. [使用pandoc转换md为PDF并添加中文支持](https://www.jianshu.com/p/7f9a9ff053bb)
4. [Pandoc+Markdown写作](https://blog.csdn.net/abcamus/article/details/52727497)