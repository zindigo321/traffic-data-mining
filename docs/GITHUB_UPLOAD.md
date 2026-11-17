# 匿名公开副本 GitHub提交

本副本已清理文件名、正文、Office/PDF作者属性及日志身份字段；旧版未匿名的作业包、ZIP和PPT不应再加入此仓库或Release。审查结果见根目录PrivacyReview.json。

## 新建仓库

打开https://github.com/new，以本人账号创建Public空仓库，例如traffic-data-mining-course。不要勾选自动生成README、.gitignore或License；本地已经准备好文件。

## 本地检查与首次提交

本副本已初始化main并设置仅本仓库生效的匿名作者名Course Archive及占位邮箱archive@example.invalid，不使用本机全局个人署名。

```powershell
Set-Location -LiteralPath 'G:\gpt-file\Codex\outputs\TrafficDataMining_GitHub'
git config --local user.name
git config --local user.email
git add .
git status --short
git diff --cached --stat
git commit -m "Archive anonymized traffic data mining coursework"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/traffic-data-mining-course.git
git push -u origin main
```

将YOUR_GITHUB_USERNAME换成本人GitHub账号，若仓库名不同也一并更改。第一次push如弹出浏览器，按提示登录授权；无需把密码或Token写入代码、URL或聊天。匿名提交字段不改变公开仓库所属的GitHub账号，仓库拥有者仍在GitHub网页上可见。

## 后续更新

添加新文件前检查正文、文件属性、截图、署名及学号，沿用本仓库匿名提交身份。新运行日志可能记录本机路径，先审查再提交。

```powershell
git status
git add assignments README.md
git diff --cached --stat
git commit -m "Update anonymized coursework"
git push
```

本副本最大单文件约22.5MB，普通Git足够。不要提交原课程压缩包、未匿名期末ZIP或个人轨迹。正式课程提交所需的实名材料另行保管，不能混入公开副本。

## 来源与复核

- https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

本次只完成本地匿名整理，没有创建远程仓库、commit或push。

匿名提交邮箱设置依据：https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address
