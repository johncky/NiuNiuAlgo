
function get_summary(update_handler) {
    var add_algo = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;

    fetch('/data').then(function (response) {
        return response.json();
    }).then(function (new_data) {
        update_handler(new_data, add_algo);
    }).catch(function (err) {
        console.log(err);
    });
}

function add_algo(update_handler) {
    var algo_ip = document.getElementById("mod_algo_ip").value;
    fetch('/add_algo?' + 'ip=' + algo_ip).then(function (response) {
        return response.json();
    }).then(function (result) {
        alert(result.response);
        get_summary(update_handler, true);
    }).catch(function () {
        alert('Failed to add algo');
    });
}

function remove_algo(update_handler) {
    var algo_ip = document.getElementById("mod_algo_ip").value;
    fetch('/remove_algo?' + 'ip=' + algo_ip).then(function (response) {
        return response.json();
    }).then(function (result) {
        alert(result.response);
        get_summary(update_handler);
    }).catch(function () {
        alert('Failed to remove algo');
    });
}