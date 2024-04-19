document.getElementById("searchTicker").addEventListener("click", function() {
    // TODO: load model

    // TODO: display loading status

    // TODO: display graphs and stuff


    // temporary code: logs button clicked, appends html to show the contents of the button when clicked
    console.log("Button clicked!");
    var mainElement = document.getElementsByTagName("main")[0];
    var ticker = document.getElementById("input-box").value.toUpperCase();
    var newElement = document.createElement("div");
    newElement.textContent = "searching data and predicting prices for " + ticker;
    mainElement.appendChild(newElement);
   
});