#!/bin/bash
### Performs a full reset of the shopping environment.
### Note: This takes a while (~2 minutes), so it's not recommended to run this too frequently.

# Define variables
CONTAINER_NAME="shopping"

docker stop $CONTAINER_NAME
docker rm $(docker ps -a | grep $CONTAINER_NAME | awk '{print $1}')
docker run --name $CONTAINER_NAME -p 7770:80 -d shopping_final_0712
# wait ~1 min for all services to start
sleep 60

docker exec $CONTAINER_NAME /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770" # no trailing slash
docker exec $CONTAINER_NAME mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path = "web/secure/base_url";'
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento cache:flush

docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_product
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_rule
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule catalogsearch_fulltext
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule catalog_category_product
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule customer_grid
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule design_config_grid
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule inventory
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_category
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_attribute
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_price
docker exec $CONTAINER_NAME /var/www/magento2/bin/magento indexer:set-mode schedule cataloginventory_stock


