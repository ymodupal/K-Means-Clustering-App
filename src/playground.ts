import './assets/styles.css';

export class EntryPoint {

    constructor() {
        console.log("constructor is called!")
    }

    showalert() {
        alert("Success!!");
    }
};

(window as any).showalert = function () {
    let hero = new EntryPoint();
    hero.showalert();
};