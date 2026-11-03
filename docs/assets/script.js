
document.addEventListener('DOMContentLoaded', function () {
  // mobile nav
  var burger = document.querySelector('.burger');
  if (burger) burger.addEventListener('click', function () { document.body.classList.toggle('nav-open'); });

  // copy buttons
  document.querySelectorAll('pre').forEach(function (pre) {
    var btn = document.createElement('button');
    btn.className = 'copy'; btn.textContent = 'Copy';
    btn.addEventListener('click', function () {
      var text = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1400);
      });
    });
    pre.appendChild(btn);
  });

  // sidebar filter
  var search = document.getElementById('navFilter');
  if (search) {
    search.addEventListener('input', function () {
      var q = search.value.toLowerCase();
      document.querySelectorAll('.sidebar a.item').forEach(function (a) {
        a.style.display = a.textContent.toLowerCase().indexOf(q) > -1 ? 'block' : 'none';
      });
      document.querySelectorAll('.sidebar h4').forEach(function (h) { h.style.display = q ? 'none' : 'block'; });
    });
  }

  // inline maths: $$...$$ -> styled monospace span
  document.querySelectorAll('.wrap p, .wrap li, .wrap td, .wrap th').forEach(function (el) {
    if (el.querySelector('code, pre')) return;
    if (el.innerHTML.indexOf('$$') === -1) return;
    el.innerHTML = el.innerHTML.replace(/\$\$([^$]+?)\$\$/g, '<span class="math">$1</span>');
  });

  // build in-page TOC
  var toc = document.getElementById('toc');
  if (toc) {
    var hs = document.querySelectorAll('.wrap h2');
    if (hs.length > 1) {
      var ol = document.createElement('ol');
      hs.forEach(function (h, i) {
        if (!h.id) h.id = 's' + i;
        var li = document.createElement('li');
        var a = document.createElement('a');
        a.href = '#' + h.id; a.textContent = h.textContent.replace(/^[0-9]+\.\s*/, '');
        li.appendChild(a); ol.appendChild(li);
      });
      toc.appendChild(ol);
      toc.style.display = 'block';
    }
  }
});
