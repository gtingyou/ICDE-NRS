var results = [];
var json_results = [];
var todaysDate = new Date();
console.log(todaysDate);

chrome.history.search({text: 'www.chinatimes.com/', maxResults: 1000,}, function(HistoryItem) {
	var cht_news_content = document.querySelector('#cht-news-content');
	var cht_news = [];

	// console.log(HistoryItem.length);
	for (var i = 0; i < HistoryItem.length; i++) {
		var item  = HistoryItem[i];
		var url   = item.url;
		var title = item.title.replace(',','')
		var visitCount = item.visitCount;
		var lastVisitTime = item.lastVisitTime;
		// convert timestamp to datetime
		var lastVisitDate = new Date(Math.floor(lastVisitTime/1000)*1000);
		
		// If browsed date equals today's date
		if(lastVisitDate.toDateString() == todaysDate.toDateString()) {
			// convert to yyyy/MM/dd/ HH:mm:ss format
			var ds =
			lastVisitDate.getFullYear() + "/" + 
			("00" + (lastVisitDate.getMonth() + 1)).slice(-2) + "/" + 
			("00" + lastVisitDate.getDate()).slice(-2) + " " +
			("00" + lastVisitDate.getHours()).slice(-2) + ":" + 
			("00" + lastVisitDate.getMinutes()).slice(-2) + ":" + 
			("00" + lastVisitDate.getSeconds()).slice(-2);
			results.push([url, title, ds]);
			json_results.push({"NewsURL":url, "NewsTitle":title, "ReadTime":ds});
			cht_news.push('<a class="list-group-item list-group-item-action" target="_blank" href="'+url+'">'+title+'<div class="row justify-content-between"><div class="col-9"><small>Last visit time: '+ds+'</small></div><div class="col-1"><small class="badge bg-primary rounded-pill">'+visitCount+'</small></div></div></a>');
		}
	}
	cht_news_content.innerHTML = cht_news.join('');
});

chrome.history.search({text: 'ltn.com.tw/', maxResults: 1000,}, function(HistoryItem) {
	var lbt_news_content = document.querySelector('#lbt-news-content');
	var lbt_news = [];
	
	// console.log(HistoryItem.length);
	for (var i = 0; i < HistoryItem.length; i++) {
		var item  = HistoryItem[i];
		var url   = item.url;
		var title = item.title.replace(',','')
		var visitCount = item.visitCount;
		var lastVisitTime = item.lastVisitTime;
		// convert timestamp to datetime
		var lastVisitDate = new Date(Math.floor(lastVisitTime/1000)*1000);

		// If browsed date equals today's date
		if(lastVisitDate.toDateString() == todaysDate.toDateString()) {
			// convert to yyyy/MM/dd/ HH:mm:ss format
			var ds =
			lastVisitDate.getFullYear() + "/" + 
			("00" + (lastVisitDate.getMonth() + 1)).slice(-2) + "/" + 
			("00" + lastVisitDate.getDate()).slice(-2) + " " +
			("00" + lastVisitDate.getHours()).slice(-2) + ":" + 
			("00" + lastVisitDate.getMinutes()).slice(-2) + ":" + 
			("00" + lastVisitDate.getSeconds()).slice(-2);
			results.push([url, title, ds]);
			json_results.push({"NewsURL":url, "NewsTitle":title, "ReadTime":ds});
			lbt_news.push('<a class="list-group-item list-group-item-action" target="_blank" href="'+url+'">'+title+'<div class="row justify-content-between"><div class="col-9"><small>last visit time: '+ds+'</small></div><div class="col-1"><small class="badge bg-primary rounded-pill">'+visitCount+'</small></div></div></a>');
		}
	}
	lbt_news_content.innerHTML = lbt_news.join('');
});

chrome.history.search({text: 'www.cna.com.tw/', maxResults: 1000,}, function(HistoryItem) {
	var cna_news_content = document.querySelector('#cna-news-content');
	var cna_news = [];
	
	// console.log(HistoryItem.length);
	for (var i = 0; i < HistoryItem.length; i++) {
		var item  = HistoryItem[i];
		var url   = item.url;
		var title = item.title.replace(',','')
		var visitCount = item.visitCount;
		var lastVisitTime = item.lastVisitTime;
		// convert timestamp to datetime
		var lastVisitDate = new Date(Math.floor(lastVisitTime/1000)*1000);

		// If browsed date equals today's date
		if(lastVisitDate.toDateString() == todaysDate.toDateString()) {
			// convert to yyyy/MM/dd/ HH:mm:ss format
			var ds =
			lastVisitDate.getFullYear() + "/" + 
			("00" + (lastVisitDate.getMonth() + 1)).slice(-2) + "/" + 
			("00" + lastVisitDate.getDate()).slice(-2) + " " +
			("00" + lastVisitDate.getHours()).slice(-2) + ":" + 
			("00" + lastVisitDate.getMinutes()).slice(-2) + ":" + 
			("00" + lastVisitDate.getSeconds()).slice(-2);
			results.push([url, title, ds]);
			json_results.push({"NewsURL":url, "NewsTitle":title, "ReadTime":ds});
			cna_news.push('<a class="list-group-item list-group-item-action" target="_blank" href="'+url+'">'+title+'<div class="row justify-content-between"><div class="col-9"><small>last visit time: '+ds+'</small></div><div class="col-1"><small class="badge bg-primary rounded-pill">'+visitCount+'</small></div></div></a>');
		}
	}
	cna_news_content.innerHTML = cna_news.join('');
});

chrome.history.search({text: 'https://www.setn.com/', maxResults: 1000,}, function(HistoryItem) {
	var setn_news_content = document.querySelector('#setn-news-content');
	var setn_news = [];
	
	// console.log(HistoryItem.length);
	for (var i = 0; i < HistoryItem.length; i++) {
		var item  = HistoryItem[i];
		var url   = item.url;
		var title = item.title.replace(',','')
		var visitCount = item.visitCount;
		var lastVisitTime = item.lastVisitTime;
		// convert timestamp to datetime
		var lastVisitDate = new Date(Math.floor(lastVisitTime/1000)*1000);

		// If browsed date equals today's date
		if(lastVisitDate.toDateString() == todaysDate.toDateString()) {
			// convert to yyyy/MM/dd/ HH:mm:ss format
			var ds =
			lastVisitDate.getFullYear() + "/" + 
			("00" + (lastVisitDate.getMonth() + 1)).slice(-2) + "/" + 
			("00" + lastVisitDate.getDate()).slice(-2) + " " +
			("00" + lastVisitDate.getHours()).slice(-2) + ":" + 
			("00" + lastVisitDate.getMinutes()).slice(-2) + ":" + 
			("00" + lastVisitDate.getSeconds()).slice(-2);
			results.push([url, title, ds]);
			json_results.push({"NewsURL":url, "NewsTitle":title, "ReadTime":ds});
			setn_news.push('<a class="list-group-item list-group-item-action" target="_blank" href="'+url+'">'+title+'<div class="row justify-content-between"><div class="col-9"><small>last visit time: '+ds+'</small></div><div class="col-1"><small class="badge bg-primary rounded-pill">'+visitCount+'</small></div></div></a>');
		}
	}
	setn_news_content.innerHTML = setn_news.join('');
});

chrome.history.search({text: 'news.tvbs.com.tw/', maxResults: 1000,}, function(HistoryItem) {
	var tvbs_news_content = document.querySelector('#tvbs-news-content');
	var tvbs_news = [];
	
	// console.log(HistoryItem.length);
	for (var i = 0; i < HistoryItem.length; i++) {
		var item  = HistoryItem[i];
		var url   = item.url;
		var title = item.title.replace(',','')
		var visitCount = item.visitCount;
		var lastVisitTime = item.lastVisitTime;
		// convert timestamp to datetime
		var lastVisitDate = new Date(Math.floor(lastVisitTime/1000)*1000);

		// If browsed date equals today's date
		if(lastVisitDate.toDateString() == todaysDate.toDateString()) {
			// convert to yyyy/MM/dd/ HH:mm:ss format
			var ds =
			lastVisitDate.getFullYear() + "/" + 
			("00" + (lastVisitDate.getMonth() + 1)).slice(-2) + "/" + 
			("00" + lastVisitDate.getDate()).slice(-2) + " " +
			("00" + lastVisitDate.getHours()).slice(-2) + ":" + 
			("00" + lastVisitDate.getMinutes()).slice(-2) + ":" + 
			("00" + lastVisitDate.getSeconds()).slice(-2);
			results.push([url, title, ds]);
			json_results.push({"NewsURL":url, "NewsTitle":title, "ReadTime":ds});
			tvbs_news.push('<a class="list-group-item list-group-item-action" target="_blank" href="'+url+'">'+title+'<div class="row justify-content-between"><div class="col-9"><small>last visit time: '+ds+'</small></div><div class="col-1"><small class="badge bg-primary rounded-pill">'+visitCount+'</small></div></div></a>');
		}
	}
	tvbs_news_content.innerHTML = tvbs_news.join('');
});

chrome.history.search({text: 'udn.com/', maxResults: 1000,}, function(HistoryItem) {
	var udn_news_content = document.querySelector('#udn-news-content');
	var udn_news = [];
	
	// console.log(HistoryItem.length);
	for (var i = 0; i < HistoryItem.length; i++) {
		var item  = HistoryItem[i];
		var url   = item.url;
		var title = item.title.replace(',','')
		var visitCount = item.visitCount;
		var lastVisitTime = item.lastVisitTime;
		// convert timestamp to datetime
		var lastVisitDate = new Date(Math.floor(lastVisitTime/1000)*1000);

		// If browsed date equals today's date
		if(lastVisitDate.toDateString() == todaysDate.toDateString()) {
			// convert to yyyy/MM/dd/ HH:mm:ss format
			var ds =
			lastVisitDate.getFullYear() + "/" + 
			("00" + (lastVisitDate.getMonth() + 1)).slice(-2) + "/" + 
			("00" + lastVisitDate.getDate()).slice(-2) + " " +
			("00" + lastVisitDate.getHours()).slice(-2) + ":" + 
			("00" + lastVisitDate.getMinutes()).slice(-2) + ":" + 
			("00" + lastVisitDate.getSeconds()).slice(-2);
			results.push([url, title, ds]);
			json_results.push({"NewsURL":url, "NewsTitle":title, "ReadTime":ds});
			udn_news.push('<a class="list-group-item list-group-item-action" target="_blank" href="'+url+'">'+title+'<div class="row justify-content-between"><div class="col-9"><small>last visit time: '+ds+'</small></div><div class="col-1"><small class="badge bg-primary rounded-pill">'+visitCount+'</small></div></div></a>');
		}
	}
	udn_news_content.innerHTML = udn_news.join('');
});




// document.getElementById("downloadButton").onclick = function(){ 
// 	downloadExcel(results)
// };

// function downloadExcel(results) {
// 	console.log(results)

// 	var data = results;
// 	var csvContent = data.map(row => row.join(",")).join("\n");
// 	var csvData = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' }); 
// 	var csvUrl = URL.createObjectURL(csvData);
// 	// chrome.downloads.download({ url: csvUrl });

// 	// filename format e.g. 2021-01-01
// 	var filename = new Date().toJSON().slice(0,10) //.replace(/-/g,'/'); 
// 	var link = document.createElement("a");
// 	link.setAttribute("href", csvUrl);
// 	link.setAttribute("download", filename+'.csv');
// 	link.style.visibility = 'hidden';
// 	console.log(link);
// 	document.body.appendChild(link);
// 	link.click();
// 	document.body.removeChild(link);
// };


document.getElementById("uploadButton").onclick = function(){ 
	upload2sql(json_results)
};

function upload2sql(json_results) {
	const user_name = document.getElementById('user-name').value
	console.log(user_name)

	data = {"user_name": user_name, "read_list": json_results}
	console.log(data)

	var url = 'http://140.114.55.4:5000/api/insert_read_list'
	// var url = 'http://203.145.219.199:55577/api/insert_read_list'
	fetch(url, {
		method: 'POST',
		body: JSON.stringify(data),
		headers: {
			'Accept': 'application/json, text/plain, */*',
			'Content-Type': 'application/json'
		}
	}).then((res) => {
		if (res.ok) {
			alert('Success')
		}
		return res.text()
	})
	.then(response => console.log(response))
	.catch(err => console.log(err))

};

