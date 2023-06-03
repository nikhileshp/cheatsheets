## Website (nikhileshp.github.io)

#### Local Setup (Standard)

Assuming you have [Ruby](https://www.ruby-lang.org/en/downloads/) and [Bundler](https://bundler.io/) installed on your system (*hint: for ease of managing ruby gems, consider using [rbenv](https://github.com/rbenv/rbenv)*).

```bash
$ bundle install
$ bundle exec jekyll serve --lsi
```

In order to check the ouput of the api call you make to notion for the blog

```bash
$ curl -X POST 'https://api.notion.com/v1/databases/66e4dc0bf46f480a903408b7725e9ecd/query' \
  -H 'Authorization: secret_tn7WHwJcnH57na08yt52eTn0yiEXA6gp7cHE3J5mtyp' \
  -H 'Notion-Version: 2022-06-28' \
  -H "Content-Type: application/json" 
```

Now, feel free to customize the theme however you like (don't forget to change the name!).
After you are done, **commit** your final changes.
