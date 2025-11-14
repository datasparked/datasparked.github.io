---
permalink: /tech/
title: "Technology"
excerpt: "A series of articles about popular tools used in tech"
last_modified_at: 2025-11-13T11:59:26-04:00
header:
  teaser: /assets/images/header_images/alexandre-debieve-FO7JIlwjOtU-unsplash.jpg
  overlay_image: /assets/images/header_images/alexandre-debieve-FO7JIlwjOtU-unsplash.jpg
  overlay_filter: 0.5
  caption: "Photo credit: [**Alexandre Debiève**](https://unsplash.com/photos/macro-photography-of-black-circuit-board-FO7JIlwjOtU?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)"
---

In today's rapidly evolving technology landscape, choosing the right tools can make or break your project's success. With over 200 powerful tools across 13 categories, developers and teams face an overwhelming array of options. This series of article analyzes the entire tech ecosystem to help you make informed decisions, optimize workflows, and build scalable, maintainable solutions.


<!-- Create array of posts with category 'cheat sheet' and sort them alphabetically -->

{% assign sortedPosts = site.categories['tech'] | sort: 'title' %}

<!-- Create a list of post using the array defined earlier -->

<ul>
  {% for post in sortedPosts %}
    {% if post.url %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

