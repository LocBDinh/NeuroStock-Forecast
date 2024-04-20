document.getElementById("searchTicker").addEventListener("click", function() {
    // Send Input to Flask App
    var ticker = document.getElementById("input-box").value.toUpperCase();
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/stock-predictions", true);
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            console.log(xhr.responseText);
            print(xhr.responseText);
        } else {
            console.log("Error: " + xhr.status);
        }
    };
    xhr.send("ticker=" + ticker);
});
