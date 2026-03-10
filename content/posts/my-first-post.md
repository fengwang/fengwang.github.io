---
title: "Hugo Qucik Introduction"
date: 2026-03-10T21:20:12+01:00
draft: false
---

### First post with Hugo

In the root folder, execute command
```bash
hugo new content content/posts/my-first-post.md
```

This will create a new markdown file in the `content/posts` directory with the name `my-first-post.md`. The file will contain some default front matter, which you can edit to include your desired title, date, and other metadata.

```
---
title: "My First Post"
date: 2026-03-10T21:20:12+01:00
draft: true
---
```

The metadata are intoduced in Hugo as "front matter". The `title` field specifies the title of the post, the `date` field indicates when the post was created, and the `draft` field is set to `true`, which means that the post will not be published until you change it to `false`. More information about front matter can be found in the [Hugo documentation](https://gohugo.io/content-management/front-matter/).


### Quick review of the post content

Execute the command
```bash
hugo server -D
```

Then open your web browser and navigate to `http://localhost:1313/`. You should see your new post listed on the homepage. Click on the post title to view the full content of the post. Since the `draft` field is set to `true`, the post will not be visible on the public site until you change it to `false` and rebuild your site.


### Publish the post

To publish the post, open the `my-first-post.md` file and change the `draft` field from `true` to `false`. Then save the file and rebuild your site by running the command:

```bash
hugo
```
After rebuilding, your post will be published and visible on the public site. You can verify this by navigating to `http://localhost:1313/` again and checking that your post is now listed on the homepage.







