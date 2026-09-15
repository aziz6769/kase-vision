from pathlib import Path

BASE = Path(__file__).resolve().parent
index = BASE / "index.html"
server = BASE / "server.py"

# Fix the ticker display name typo introduced during deployment preparation.
if server.exists():
    text = server.read_text(encoding="utf-8")
    text = text.replace('"KEGC":"KEGC"', '"KEGC":"KEGOC"')
    server.write_text(text, encoding="utf-8")

# Add a small, independent UI repair layer. It uses the existing API and
# fills the two tables that can remain empty if the original frontend JS
# does not call its table renderer after the analysis is loaded.
patch = r'''<script id="kase-vision-ui-repair">
(function(){
  const esc = v => String(v).replace(/[&<>\"]/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[s]));
  const pct = v => (Number(v) * 100).toFixed(2) + '%';
  let repairPage = 1;

  async function getAnalysis(){
    const r = await fetch('/api/analysis');
    if(!r.ok) throw new Error('analysis '+r.status);
    return r.json();
  }

  async function renderWeights(key){
    const box = document.getElementById('weights');
    if(!box) return;
    try{
      const data = await getAnalysis();
      const z = data.result.portfolios[key] || data.result.portfolios.max_sharpe;
      box.innerHTML = '<tr><th>Актив</th><th>Доля</th></tr>' +
        Object.entries(z.weights).map(([t,w]) => '<tr><td>'+esc(t)+'</td><td>'+pct(w)+'</td></tr>').join('');
    }catch(e){
      box.innerHTML = '<tr><td>Не удалось загрузить структуру</td></tr>';
      console.error(e);
    }
  }

  async function renderVariants(p){
    repairPage = Math.max(1, Number(p)||1);
    const table = document.getElementById('variants');
    if(!table) return;
    const sort = document.getElementById('sort')?.value || 'sharpe';
    const direction = document.getElementById('dir')?.value || 'desc';
    try{
      const r = await fetch('/api/portfolios?page='+repairPage+'&limit=25&sort='+encodeURIComponent(sort)+'&direction='+encodeURIComponent(direction));
      if(!r.ok) throw new Error('portfolios '+r.status);
      const data = await r.json();
      table.innerHTML = '<tr><th>#</th><th>Доходность</th><th>Риск</th><th>Sharpe</th></tr>' +
        data.rows.map(x => '<tr><td>'+x.id+'</td><td>'+pct(x.return)+'</td><td>'+pct(x.risk)+'</td><td>'+Number(x.sharpe).toFixed(2)+'</td></tr>').join('');
      const info = document.getElementById('pageInfo');
      if(info) info.textContent = 'Страница '+data.page+' • '+data.total.toLocaleString('ru-RU')+' портфелей';
      const prev = document.getElementById('prev'), next = document.getElementById('next');
      if(prev) prev.disabled = data.page <= 1;
      if(next) next.disabled = data.page * data.limit >= data.total;
    }catch(e){
      table.innerHTML = '<tr><td>Не удалось загрузить варианты портфелей</td></tr>';
      console.error(e);
    }
  }

  // Override the inline pagination callback used by the existing page.
  window.loadVariants = renderVariants;

  window.addEventListener('load', function(){
    setTimeout(function(){
      renderWeights('max_sharpe');
      renderVariants(1);
      document.querySelectorAll('.strategy button[data-k]').forEach(btn => {
        btn.addEventListener('click', () => renderWeights(btn.dataset.k));
      });
      document.getElementById('sort')?.addEventListener('change', () => renderVariants(1));
      document.getElementById('dir')?.addEventListener('change', () => renderVariants(1));
    }, 250);
  });
})();
</script>'''

if index.exists():
    text = index.read_text(encoding="utf-8")
    marker = '<script id="kase-vision-ui-repair">'
    if marker not in text:
        if '</body>' not in text:
            raise SystemExit('index.html: </body> not found')
        text = text.replace('</body>', patch + '\n</body>')
        index.write_text(text, encoding="utf-8")
        print('KASE Vision UI repair applied.')
    else:
        print('KASE Vision UI repair already present.')
else:
    raise SystemExit('index.html not found')
