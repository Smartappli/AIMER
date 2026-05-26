document.addEventListener("submit", (event) => {
  const submitter = event.submitter;
  if (!(submitter instanceof HTMLElement)) {
    return;
  }

  const text = submitter.querySelector("#btnText");
  const loader = submitter.querySelector("#btnLoader");
  if (text instanceof HTMLElement) {
    text.classList.add("visually-hidden");
  }
  if (loader instanceof HTMLElement) {
    loader.classList.remove("visually-hidden");
  }
});
