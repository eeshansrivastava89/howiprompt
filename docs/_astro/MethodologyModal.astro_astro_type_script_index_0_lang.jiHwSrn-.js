(function(){const c="/images/",v=[{key:"commander",img:c+"char_commander.png",name:"The Commander",badge:"Brief + Directive",trait:"Direct, efficient, decisive",accent:"#41608e",accentDark:"#7da4d4",gradient:"linear-gradient(160deg,#d4dde8,#c2cedd)",gradientDark:"linear-gradient(160deg,#222830,#1a2028)",desc:"You give precise orders in short bursts. No wasted words, no hand-holding — you know exactly what you want and expect the AI to deliver it on the first try.",prompt:"“Patch this regression, keep the API stable, and show me the exact files changed.”",liner:"You prefer decisive progress over collaborative wandering.",axes:{"Detail Level":30,"Communication Style":25}},{key:"partner",img:c+"char_partner.png",name:"The Partner",badge:"Brief + Collaborative",trait:"Conversational, collaborative, adaptive",accent:"#8a5d73",accentDark:"#c48aa8",gradient:"linear-gradient(160deg,#e8d8e0,#ddc8d4)",gradientDark:"linear-gradient(160deg,#2e2428,#261c22)",desc:"You keep it short but keep it human. Quick exchanges, polite nudges, and a conversational flow that treats AI as a co-pilot, not a tool.",prompt:"“Better, but not there yet. Keep the tone, reduce complexity, and give me a second version with a stronger opening.”",liner:"You keep pushing until the output finally clicks.",axes:{"Detail Level":30,"Communication Style":75}},{key:"architect",img:c+"char_architect.png",name:"The Architect",badge:"Detailed + Directive",trait:"Precise, methodical, thorough",accent:"#7b5a42",accentDark:"#c4a07a",gradient:"linear-gradient(160deg,#e8ddd2,#ddd0c2)",gradientDark:"linear-gradient(160deg,#342c24,#2a2218)",desc:"You write specs, not prompts. Every request comes with constraints, file paths, and numbered requirements. You leave nothing to chance.",prompt:"“Use the existing component API. Keep the interaction model unchanged. Add keyboard support, tests, and a migration note.”",liner:"You like the model to be powerful, but not unsupervised.",axes:{"Detail Level":80,"Communication Style":25}},{key:"explorer",img:c+"char_explorer.png",name:"The Explorer",badge:"Detailed + Collaborative",trait:"Curious, analytical, open-minded",accent:"#3f7a6e",accentDark:"#6abfae",gradient:"linear-gradient(160deg,#d4e8e3,#c2ddd6)",gradientDark:"linear-gradient(160deg,#222e2b,#1a2624)",desc:"You bring context and ask questions. Every conversation is an investigation — you explain what you’re thinking, ask for alternatives, and dig into the details together.",prompt:"“Compare these three approaches, surface tradeoffs, and tell me what an experienced engineer would worry about.”",liner:"You use the model as an idea space, not just a task runner.",axes:{"Detail Level":80,"Communication Style":75}}],y=()=>document.documentElement.classList.contains("dark");let l="commander";const m=document.getElementById("methPersonaList"),s=document.getElementById("methPersonaDetail");if(!m||!s)return;function h(){m.innerHTML=v.map(e=>`
      <button class="meth-p-thumb ${e.key===l?"active":""}" data-pkey="${e.key}"
              style="--pa:${y()?e.accentDark:e.accent}">
        <div class="meth-p-thumb-img"><img src="${e.img}" alt="${e.name}"/></div>
        <div class="meth-p-thumb-text">
          <div class="meth-p-thumb-name">${e.name.replace("The ","")}</div>
          <div class="meth-p-thumb-trait">${e.trait}</div>
        </div>
      </button>`).join("")}function g(e,a){const t=v.find(o=>o.key===e);if(!t)return;const n=y()?t.accentDark:t.accent;s.style.setProperty("--pa",n),s.style.setProperty("--pd-gradient",t.gradient),s.style.setProperty("--pd-gradient-dark",t.gradientDark),s.innerHTML=`
      <div class="meth-pd-art">
        <img src="${t.img}" alt="${t.name}" style="opacity:0;transform:scale(.9)"/>
      </div>
      <div class="meth-pd-copy">
        <div class="meth-pd-badge" style="color:${n}">${t.badge}</div>
        <div class="meth-pd-name">${t.name}</div>
        <p class="meth-pd-desc">${t.desc}</p>
        <div class="meth-pd-cards">
          <div class="meth-pd-card">
            <div class="meth-pd-card-label">Signature prompt</div>
            <div class="meth-pd-card-text">${t.prompt}</div>
          </div>
          <div class="meth-pd-card">
            <div class="meth-pd-card-label">In a nutshell</div>
            <div class="meth-pd-card-text">${t.liner}</div>
          </div>
        </div>
        <div class="meth-pd-axes">
          ${Object.entries(t.axes).map(([o,u])=>`
            <div class="meth-ax">
              <div class="meth-ax-label"><span>${o}</span><span>${u}</span></div>
              <div class="meth-ax-track"><div class="meth-ax-fill" style="width:0%;background:${n}"></div></div>
            </div>`).join("")}
        </div>
      </div>`,requestAnimationFrame(()=>{const o=s.querySelector(".meth-pd-art img");o&&(o.style.opacity="1",o.style.transform="scale(1)"),s.querySelectorAll(".meth-ax-fill").forEach((u,x)=>{const E=Object.values(t.axes)[x];setTimeout(()=>u.style.width=E+"%",50+x*60)})})}function w(e){l=e,h(),g(e)}m.addEventListener("click",e=>{const a=e.target.closest("[data-pkey]");a&&w(a.dataset.pkey)}),h(),g("commander"),new MutationObserver(()=>{h(),g(l)}).observe(document.documentElement,{attributes:!0,attributeFilter:["class"]});const i=document.getElementById("methScroll");if(!i)return;const b=new IntersectionObserver(e=>{e.forEach(a=>{a.isIntersecting&&a.target.classList.add("vis")})},{threshold:.1,root:i});i.querySelectorAll(".meth-reveal").forEach(e=>b.observe(e));const f=i.querySelectorAll("[data-meth-section]"),p=document.querySelectorAll(".meth-dot"),d=document.getElementById("methProgressFill");i.addEventListener("scroll",()=>{const e=i.scrollHeight-i.clientHeight;d&&(d.style.width=(e>0?i.scrollTop/e*100:0)+"%");let a=0;const t=i.getBoundingClientRect();f.forEach((r,n)=>{r.getBoundingClientRect().top-t.top<t.height*.5&&(a=n)}),p.forEach((r,n)=>r.classList.toggle("active",n===a))}),p.forEach(e=>e.addEventListener("click",()=>{const a=parseInt(e.dataset.methIdx);f[a]?.scrollIntoView({behavior:"smooth"})}));const k=document.getElementById("methodologyModal");k&&new MutationObserver(a=>{for(const t of a)t.target.classList.contains("active")&&(i.scrollTop=0,d&&(d.style.width="0%"),p.forEach((r,n)=>r.classList.toggle("active",n===0)),i.querySelectorAll(".meth-reveal").forEach(r=>r.classList.remove("vis")),setTimeout(()=>{i.querySelectorAll(".meth-reveal").forEach(r=>b.observe(r))},50))}).observe(k,{attributes:!0,attributeFilter:["class"]})})();
