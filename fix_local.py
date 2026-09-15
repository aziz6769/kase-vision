from pathlib import Path

BASE = Path(__file__).resolve().parent
index = BASE / "index.html"
server = BASE / "server.py"

if server.exists():
    text = server.read_text(encoding="utf-8")
    text = text.replace('"KEGC":"KEGC"', '"KEGC":"KEGOC"')
    server.write_text(text, encoding="utf-8")

if not index.exists():
    raise SystemExit('index.html not found')

text = index.read_text(encoding="utf-8")
text = text.replace('<section id="variants" class="card">', '<section id="variantsSection" class="card">')
text = text.replace('<a href="#variants">Варианты</a>', '<a href="#variantsSection">Варианты</a>')

# Replace any previous repair layer with the current production-safe layer.
marker = '<script id="kase-vision-ui-repair">'
if marker in text:
    start = text.index(marker)
    end = text.index('</script>', start) + len('</script>')
    text = text[:start] + text[end:]

patch = r'''<script id="kase-vision-ui-repair">
(function(){
  const esc=v=>String(v).replace(/[&<>\"]/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[s]));
  const pct=v=>(Number(v)*100).toFixed(2)+'%';

  async function getAnalysis(){
    const r=await fetch('/api/analysis',{cache:'no-store'});
    if(!r.ok) throw new Error('analysis '+r.status);
    return r.json();
  }

  async function renderWeights(key){
    const box=document.getElementById('weights');
    if(!box) return;
    try{
      const data=await getAnalysis();
      const portfolios=data.result && data.result.portfolios;
      if(!portfolios) throw new Error('portfolios missing');
      const z=portfolios[key]||portfolios.max_sharpe;
      if(!z || !z.weights) throw new Error('weights missing');
      box.innerHTML='<tr><th>Актив</th><th>Доля</th></tr>'+Object.entries(z.weights).map(([t,w])=>'<tr><td>'+esc(t)+'</td><td>'+pct(w)+'</td></tr>').join('');
    }catch(e){
      box.innerHTML='<tr><td colspan="2">Не удалось загрузить структуру</td></tr>';
      console.error('KASE Vision weights:',e);
    }
  }

  async function renderVariants(p){
    window.page=Math.max(1,Number(p)||1);
    const table=document.querySelector('#variantsSection table#variants');
    if(!table) return;
    const sort=document.getElementById('sort')?.value||'sharpe';
    const direction=document.getElementById('dir')?.value||'desc';
    try{
      const r=await fetch('/api/portfolios?page='+window.page+'&limit=25&sort='+encodeURIComponent(sort)+'&direction='+encodeURIComponent(direction),{cache:'no-store'});
      if(!r.ok) throw new Error('portfolios '+r.status);
      const data=await r.json();
      table.innerHTML='<tr><th>#</th><th>Доходность</th><th>Риск</th><th>Sharpe</th></tr>'+data.rows.map(x=>'<tr><td>'+x.id+'</td><td>'+pct(x.return)+'</td><td>'+pct(x.risk)+'</td><td>'+Number(x.sharpe).toFixed(2)+'</td></tr>').join('');
      const info=document.getElementById('pageInfo');
      if(info) info.textContent='Страница '+data.page+' • '+Number(data.total).toLocaleString('ru-RU')+' портфелей';
      const prev=document.getElementById('prev'),next=document.getElementById('next');
      if(prev) prev.disabled=data.page<=1;
      if(next) next.disabled=data.page*data.limit>=data.total;
    }catch(e){
      table.innerHTML='<tr><td colspan="4">Не удалось загрузить варианты портфелей</td></tr>';
      console.error('KASE Vision variants:',e);
    }
  }

  // Keep the global page used by the original inline buttons in sync.
  window.loadVariants=function(p){
    window.page=Math.max(1,Number(p)||1);
    return renderVariants(window.page);
  };

  function boot(){
    setTimeout(function(){
      renderWeights(window.selected||'max_sharpe');
      renderVariants(window.page||1);
      document.querySelectorAll('.strategy button[data-k]').forEach(btn=>{
        btn.addEventListener('click',()=>renderWeights(btn.dataset.k));
      });
      document.getElementById('sort')?.addEventListener('change',()=>renderVariants(1));
      document.getElementById('dir')?.addEventListener('change',()=>renderVariants(1));
    },700);
  }
  if(document.readyState==='loading') window.addEventListener('DOMContentLoaded',boot); else boot();
})();
</script>'''

text=text.replace('</body>',patch+'\n</body>')
index.write_text(text,encoding='utf-8')
print('KASE Vision production UI repair applied.')
