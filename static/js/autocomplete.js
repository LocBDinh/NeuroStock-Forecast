const resultsBox = document.querySelector('.result-box');
                    const inputBox = document.getElementById('input-box');
            
                    inputBox.onkeyup = function () {
                        let result = [];
                        let input = inputBox.value;
                        if (input.length) {
                            fetch('/data/tickers')
                                .then(response => {
                                    if (!response.ok) {
                                        throw new Error('Network response was not ok');
                                    }
                                    return response.json();
                                })
                                .then(data => {
                                    result = data.filter((keyword) => {
                                        return keyword.ticker.toLowerCase().includes(input.toLowerCase());
                                    });
                                    display(result);
                                })
                                .catch(error => {
                                    console.error('Error fetching JSON:', error);
                                });
                        } else {
                            result = []
                            display(result);
                        }
                    };
            
                    /**
                     * Display the autocomplete results in the results box.
                     * @param {Array} result - The array of autocomplete results.
                     */
                    function display(result) {
                        const content = result.map((list) => {
                            return "<li onclick=selectInput(this)>" +
                                "<div class='ac-ticker'>" + list.ticker + "</div>" +
                                "<div class='ac-name'>" + list.name + "</div>" +
                                "</li>";
                        });
                        if (result.length){
                            resultsBox.innerHTML = "<ul>" + content.join("") + "</ul>";
                        } else {
                            resultsBox.innerHTML = "";
                        }
                    }
            
                    function selectInput(list) {
                        inputBox.value = list.querySelector('.ac-ticker').innerHTML;
                        resultsBox.innerHTML = '';
                    }