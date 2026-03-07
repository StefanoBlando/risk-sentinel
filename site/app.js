const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  { threshold: 0.18 }
);

document.querySelectorAll(".reveal").forEach(el => observer.observe(el));

document.querySelectorAll(".metric[data-count]").forEach(node => {
  const target = Number(node.dataset.count);
  if (!Number.isFinite(target)) return;

  const duration = 900;
  const start = performance.now();

  const tick = now => {
    const p = Math.min((now - start) / duration, 1);
    const value = Math.floor(target * p).toLocaleString("en-US");
    node.textContent = value;
    if (p < 1) requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
});
