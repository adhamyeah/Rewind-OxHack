const answerEl = document.getElementById("answer");

function say(text) {
  answerEl.textContent = text;
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

async function ask(question) {
  try {
    const res = await fetch("/where?q=" + encodeURIComponent(question));
    say((await res.json()).answer);
  } catch {
    say("Sorry, I can't reach Rewind right now.");
  }
}

async function refresh() {
  let items;
  try {
    items = await (await fetch("/items")).json();
  } catch {
    return;   // server restarting: keep the tiles we have
  }
  const grid = document.getElementById("grid");
  grid.replaceChildren();
  for (const i of items) {
    const tile = document.createElement("button");
    tile.className = "tile";
    if (i.photo) {
      const img = document.createElement("img");
      img.src = "/" + i.photo;
      img.alt = `Photo of the ${i.item}`;
      tile.appendChild(img);
    }
    const name = document.createElement("h2");
    name.textContent = i.item;
    const where = document.createElement("p");
    where.textContent = `${i.status}, ${i.ago}`;
    tile.append(name, where);
    tile.onclick = () => say(i.answer);
    grid.appendChild(tile);
  }
}
refresh();
setInterval(refresh, 3000);

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const typed = document.getElementById("typed");

typed.onsubmit = e => {
  e.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (q) ask(q);
};

document.getElementById("ask").onclick = () => {
  if (!Recognition) {             // e.g. Firefox: type instead
    typed.hidden = false;
    document.getElementById("q").focus();
    return;
  }
  const rec = new Recognition();
  rec.lang = "en-GB";
  rec.onresult = e => ask(e.results[0][0].transcript);
  rec.onerror = () => {           // mic blocked or nothing heard: offer typing
    say("Sorry, I didn't hear that. You can type your question instead.");
    typed.hidden = false;
  };
  answerEl.textContent = "Listening...";
  rec.start();
};
