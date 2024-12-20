
const URL_PREFIX = document.querySelector('meta[name="hexz-url-prefix"]')?.content || "";

const nameSuggestions = [];

async function suggestUsername() {
    if (nameSuggestions.length > 0) {
        return nameSuggestions.pop();
    }
    const resp = await fetch(`${URL_PREFIX}/loginnames`);
    if (!resp.ok) {
        console.error("Could not fetch usernames: ", resp.statusText);
        return "";
    }
    const result = await resp.json();
    nameSuggestions.push(...result.names);
    if (nameSuggestions.length == 0) {
        return "";
    }
    return nameSuggestions.pop();
}

async function updateUsername() {
    const name = document.querySelector("#name");
    name.value = await suggestUsername();
}

async function initLogin() {
    const refresh = document.querySelector("#name-refresh");
    if (refresh) {
        refresh.addEventListener("click", (e) => {
            updateUsername();
        });
    }
    const input = document.querySelector("#name");
    if (input) {
        input.addEventListener("focus", () => {
            const textLength = input.value.length;
            input.select();
        })
    }
    const toggleGuest = document.querySelector("#toggle-guest");
    if (toggleGuest) {
        toggleGuest.addEventListener("click", () => {
            document.querySelector("#account-login").classList.add("hidden");
            document.querySelector("#guest-login").classList.remove("hidden");
        })
    }
    const toggleAccount = document.querySelector("#toggle-account");
    if (toggleAccount) {
        toggleAccount.addEventListener("click", () => {
            document.querySelector("#account-login").classList.remove("hidden");
            document.querySelector("#guest-login").classList.add("hidden");
        })
    }
    
    updateUsername();
}
